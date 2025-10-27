#include "Planner.hpp"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_ros/GridMapRosConverter.hpp>

std::vector<grid_map::Index> Planner::planFromPosition(const grid_map::GridMap& gridMap,
                                                       const std::string& layerName,
                                                       const grid_map::Position& startPos,
                                                       const grid_map::Position& goalPos)
{
  grid_map::Index startIndex, goalIndex;

  // Convert world positions to grid indices
  bool gotStart = gridMap.getIndex(startPos, startIndex);
  bool gotGoal  = gridMap.getIndex(goalPos, goalIndex);

  if (!gotStart) {
    ROS_WARN("Planner::planFromPosition() - Could not convert startPos to an index! startPos: %.2f, %.2f", startPos[0], startPos[1]);
    return {};
  }
  if (!gotGoal) {
    ROS_WARN("Planner::planFromPosition() - Could not convert goalPos to an index! goalPos: %.2f, %.2f", goalPos[0], goalPos[1]);
    return {};
  }

  return plan(gridMap, layerName, startIndex, goalIndex);
}
