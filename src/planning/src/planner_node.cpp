#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>
#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// Your planner interfaces
#include "Planner.hpp"
#include "AStarPlanner.hpp"

namespace {

inline double yawFromQuaternion(double x, double y, double z, double w) {
  // yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
  const double siny_cosp = 2.0 * (w*z + x*y);
  const double cosy_cosp = 1.0 - 2.0 * (y*y + z*z);
  return std::atan2(siny_cosp, cosy_cosp);
}

inline void worldToBody(double dx_w, double dy_w, double yaw, double& dx_b, double& dy_b) {
  const double c = std::cos(yaw), s = std::sin(yaw);
  dx_b =  c*dx_w + s*dy_w;   // +x forward
  dy_b = -s*dx_w + c*dy_w;   // +y left
}

inline void bodyToWorld(double dx_b, double dy_b, double yaw, double& dx_w, double& dy_w) {
  const double c = std::cos(yaw), s = std::sin(yaw);
  dx_w = c*dx_b - s*dy_b;
  dy_w = s*dx_b + c*dy_b;
}

} // namespace

class RobotCentricPlannerNode {
public:
  RobotCentricPlannerNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh)
  {
    // Params
    pnh_.param<std::string>("cost_input_layer", costInputLayer_, "traversability2");
    pnh_.param<std::string>("cost_layer",       costLayerName_,   "cost");
    pnh_.param<bool>("clamp_to_local_map", clampToLocalMap_, true);
    pnh_.param<int>("edge_margin_cells", edgeMarginCells_, 2);

    // Planner (use your existing implementation)
    planner_.reset(new AStarPlanner());

    odomSub_ = nh_.subscribe("/carla/ego_vehicle/odometry", 1, &RobotCentricPlannerNode::odomCb, this);
    mapSub_  = nh_.subscribe("/elevation_mapping1/elevation_map_raw", 1, &RobotCentricPlannerNode::mapCb, this);
    goalSub_ = nh_.subscribe("/move_base_simple/goal", 1, &RobotCentricPlannerNode::goalCb, this);

    pathPub_ = nh_.advertise<nav_msgs::Path>("/planned_path", 1, /*latch=*/true);

    ROS_INFO("Robot-Centric Planner (C++) initialized.");
  }

private:
  // ---------- Callbacks ----------
  void odomCb(const nav_msgs::OdometryConstPtr& msg) {
    odom_ = *msg;
    worldFrame_ = msg->header.frame_id; // publish paths in this frame
  }

  void mapCb(const grid_map_msgs::GridMapConstPtr& msg) {
    // Convert ROS msg -> GridMap
    grid_map::GridMap tmp;
    grid_map::GridMapRosConverter::fromMessage(*msg, tmp);

    // Prepare/refresh the 'cost' layer from the input traversability layer.
    if (!tmp.exists(costInputLayer_)) {
      ROS_WARN_THROTTLE(5.0, "Layer '%s' not in GridMap.", costInputLayer_.c_str());
      return;
    }

    if (!tmp.exists(costLayerName_)) {
      tmp.add(costLayerName_);
    }

    const auto& trav = tmp.get(costInputLayer_);
    auto& cost = tmp.get(costLayerName_);

    // cost = (1 - traversability) * 100; NaN -> +inf
    cost.setZero();
    for (grid_map::GridMapIterator it(tmp); !it.isPastEnd(); ++it) {
        const grid_map::Index idx = *it;

        const float v = tmp.at(costInputLayer_, idx);   // read input layer cell
        float& c      = tmp.at(costLayerName_,  idx);   // write cost layer cell

        if (std::isnan(v)) {
            c = std::numeric_limits<float>::infinity();
        } else {
            const float cv = (1.f - v) * 100.f;
            c = (cv >= 0.f) ? cv : 0.f;
        }
    }

    gridMap_ = std::move(tmp);
  }

  void goalCb(const geometry_msgs::PoseStampedConstPtr& msg) {
    // Preconditions
    if (!odomReceived()) {
      ROS_WARN("No odometry received. Cannot plan.");
      publishEmptyPath();
      return;
    }
    if (gridMap_.getLayers().empty() || !gridMap_.exists(costLayerName_)) {
      ROS_WARN("No valid grid map or cost layer. Cannot plan.");
      publishEmptyPath();
      return;
    }

    // Frames (warning only; we avoid TF and assume coincident origins)
    if (!msg->header.frame_id.empty() && !worldFrame_.empty() && msg->header.frame_id != worldFrame_) {
      ROS_WARN("Goal frame '%s' != world frame '%s'. Proceeding as if coincident (no TF).",
               msg->header.frame_id.c_str(), worldFrame_.c_str());
    }

    // Robot pose (world)
    const double rx = odom_.pose.pose.position.x;
    const double ry = odom_.pose.pose.position.y;
    const auto& q = odom_.pose.pose.orientation;
    const double yaw = yawFromQuaternion(q.x, q.y, q.z, q.w);

    // Goal (world)
    const double gx = msg->pose.position.x;
    const double gy = msg->pose.position.y;

    ROS_INFO("Received goal in '%s'. Robot(x=%.2f,y=%.2f,yaw=%.1f deg), Goal(x=%.2f,y=%.2f)",
             msg->header.frame_id.c_str(), rx, ry, yaw*180.0/M_PI, gx, gy);

    // World -> body (ego) goal vector
    double goal_x_b = 0.0, goal_y_b = 0.0;
    worldToBody(gx - rx, gy - ry, yaw, goal_x_b, goal_y_b);
    ROS_INFO("Goal in ego frame: x=%.2f m (forward), y=%.2f m (left)", goal_x_b, goal_y_b);

    // Grid geometry
    const double res = gridMap_.getResolution();
    const grid_map::Size size = gridMap_.getSize(); // (rows, cols)
    const int rows = static_cast<int>(size(0));
    const int cols = static_cast<int>(size(1));
    if (rows <= 0 || cols <= 0) {
      ROS_WARN("Invalid GridMap size.");
      publishEmptyPath();
      return;
    }

    // Start index: assume robot is at the geometric center
    grid_map::Index startIdx(rows/2, cols/2);

    // Ego (meters) -> grid offsets (NumPy-style): +forward => -row, +left => -col
    double dx_cells = -goal_x_b / res;
    double dy_cells = -goal_y_b / res;

    // Optionally clamp to local map
    if (clampToLocalMap_) {
      const int max_r = std::min(startIdx.x(), rows - 1 - startIdx.x()) - edgeMarginCells_;
      const int max_c = std::min(startIdx.y(), cols - 1 - startIdx.y()) - edgeMarginCells_;
      const int max_radius = std::max(0, std::min(max_r, max_c));
      const double dist_cells = std::hypot(dx_cells, dy_cells);
      if (max_radius > 0 && dist_cells > static_cast<double>(max_radius)) {
        const double scale = static_cast<double>(max_radius) / std::max(1e-6, dist_cells);
        dx_cells *= scale;
        dy_cells *= scale;
        ROS_WARN("Goal outside local map. Clamped to ~%.1f m in same direction.",
                 max_radius * res);
      }
    }

    grid_map::Index goalIdx(
      static_cast<int>(std::round(startIdx.x() + dx_cells)),
      static_cast<int>(std::round(startIdx.y() + dy_cells))
    );

    ROS_INFO("Goal offset (cells): rows=%d, cols=%d",
             static_cast<int>(std::round(dx_cells)),
             static_cast<int>(std::round(dy_cells)));
    ROS_INFO("Planning from grid index (%d,%d) to (%d,%d)",
             startIdx.x(), startIdx.y(), goalIdx.x(), goalIdx.y());

    // Bounds check
    if (!isInBounds(goalIdx, rows, cols)) {
      ROS_WARN("Goal index out of bounds. rows=%d cols=%d", rows, cols);
      publishEmptyPath();
      return;
    }

    // Call the planner over the 'cost' layer
    std::vector<grid_map::Index> pathIdx =
        planner_->plan(gridMap_, costLayerName_, startIdx, goalIdx);

    if (pathIdx.empty()) {
      ROS_WARN("Planner returned no path.");
      publishEmptyPath();
      return;
    }

    // Build and publish Path in world frame (manual ego->world transform)
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = worldFrame_.empty() ? "map" : worldFrame_;

    const int center_r = rows / 2;
    const int center_c = cols / 2;

    for (const auto& ic : pathIdx) {
      const int r = ic.x();
      const int c = ic.y();

      const int row_off = center_r - r;
      const int col_off = center_c - c;
      const double x_ego = row_off * res; // +forward
      const double y_ego = col_off * res; // +left

      double dxw = 0.0, dyw = 0.0;
      bodyToWorld(x_ego, y_ego, yaw, dxw, dyw);
      const double wx = rx + dxw;
      const double wy = ry + dyw;

      geometry_msgs::PoseStamped p;
      p.header = path.header;
      p.pose.position.x = wx;
      p.pose.position.y = wy;
      p.pose.orientation.w = 1.0; // no heading for path points
      path.poses.push_back(std::move(p));
    }

    pathPub_.publish(path);
    ROS_INFO("Path with %zu poses published.", path.poses.size());
  }

  // ---------- Helpers ----------
  static bool isInBounds(const grid_map::Index& idx, int rows, int cols) {
    return (0 <= idx.x() && idx.x() < rows && 0 <= idx.y() && idx.y() < cols);
  }

  bool odomReceived() const {
    return !worldFrame_.empty(); // set when first odom arrives
  }

  void publishEmptyPath() {
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = worldFrame_.empty() ? "map" : worldFrame_;
    pathPub_.publish(path);
  }

private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber odomSub_, mapSub_, goalSub_;
  ros::Publisher pathPub_;

  nav_msgs::Odometry odom_;
  std::string worldFrame_;

  grid_map::GridMap gridMap_;
  std::string costInputLayer_{"traversability2"};
  std::string costLayerName_{"cost"};

  bool clampToLocalMap_{true};
  int edgeMarginCells_{2};

  std::unique_ptr<Planner> planner_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "robot_centric_planner_cpp");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  RobotCentricPlannerNode node(nh, pnh);
  ros::spin();
  return 0;
}
