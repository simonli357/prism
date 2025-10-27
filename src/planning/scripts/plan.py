#!/usr/bin/env python3
import rospy
import numpy as np
import heapq
import math

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from grid_map_msgs.msg import GridMap as GridMapMsg
import time

def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_body(dx_w, dy_w, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    dx_b =  c * dx_w + s * dy_w
    dy_b = -s * dx_w + c * dy_w
    return dx_b, dy_b


def body_to_world(dx_b, dy_b, yaw):
    """
    Transform a vector from body (ego) frame to world frame.
    world = R_body_to_world @ body
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    dx_w = c * dx_b - s * dy_b
    dy_w = s * dx_b + c * dy_b
    return dx_w, dy_w

class BiAStarPlanner:
    def __init__(self, heuristic_weight=1.1, use_8conn=True):
        self.w = float(heuristic_weight)
        self.neigh = [(-1,0),(1,0),(0,-1),(0,1)]
        if use_8conn:
            self.neigh += [(-1,-1),(-1,1),(1,-1),(1,1)]

    @staticmethod
    def _heur(a, b):
        # Euclidean on grid indices
        return math.hypot(a[0]-b[0], a[1]-b[1])

    @staticmethod
    def _crop_roi(start, goal, rows, cols, margin):
        r0 = min(start[0], goal[0]) - margin
        r1 = max(start[0], goal[0]) + margin
        c0 = min(start[1], goal[1]) - margin
        c1 = max(start[1], goal[1]) + margin
        r0 = max(0, r0); c0 = max(0, c0)
        r1 = min(rows-1, r1); c1 = min(cols-1, c1)
        # expand to full bounds if degenerate
        return (r0, r1, c0, c1)

    @staticmethod
    def _reconstruct(came_from, meet, start, goal, came_from_rev=None):
        # Forward part
        fpath = [meet]
        cur = meet
        while cur in came_from:
            cur = came_from[cur]
            fpath.append(cur)
        fpath.reverse()  # start -> meet

        if came_from_rev is None:
            return fpath

        # Backward part (from meet to goal using reverse tree)
        bpath = []
        cur = meet
        while cur in came_from_rev:
            cur = came_from_rev[cur]
            bpath.append(cur)
        # fpath already includes meet; append the rest to goal
        return fpath + bpath

    def plan(self, cost_map, start, goal,
             timeout_sec=0.30,
             max_expansions=80000,
             roi_margin_cells=60):
        rows, cols = cost_map.shape

        # Basic checks
        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            rospy.logwarn("BiA*: start out of bounds."); return None
        if not (0 <= goal[0]  < rows and 0 <= goal[1]  < cols):
            rospy.logwarn("BiA*: goal out of bounds."); return None
        if not np.isfinite(cost_map[start]):
            rospy.logwarn("BiA*: start is an obstacle."); return None
        if not np.isfinite(cost_map[goal]):
            rospy.logwarn("BiA*: goal is an obstacle."); return None

        r0,r1,c0,c1 = self._crop_roi(start, goal, rows, cols, roi_margin_cells)
        sub = cost_map[r0:r1+1, c0:c1+1]
        H, W = sub.shape

        s = (start[0]-r0, start[1]-c0)
        g = (goal[0]-r0,  goal[1]-c0)

        inf = np.inf
        gF = np.full((H,W), inf, dtype=np.float32)
        gB = np.full((H,W), inf, dtype=np.float32)
        fF = np.full((H,W), inf, dtype=np.float32)
        fB = np.full((H,W), inf, dtype=np.float32)
        closedF = np.zeros((H,W), dtype=bool)
        closedB = np.zeros((H,W), dtype=bool)

        gF[s] = 0.0
        gB[g] = 0.0
        fF[s] = self.w * self._heur(s, g)
        fB[g] = self.w * self._heur(g, s)

        openF = [(fF[s], 0, s)]
        openB = [(fB[g], 0, g)]
        push_idF = 1
        push_idB = 1

        cameF = {}
        cameB = {}

        best_meet = None
        best_cost = inf

        t0 = time.monotonic()
        expansions = 0

        def expand(openQ, g_this, f_this, g_other, closed_this, came_this,
                   target, push_id, forward=True):
            nonlocal best_meet, best_cost, expansions

            if not openQ:
                return push_id, False

            _, _, current = heapq.heappop(openQ)
            if closed_this[current]:
                return push_id, True
            closed_this[current] = True
            expansions += 1

            cr, cc = current
            if closedB[cr, cc] if forward else closedF[cr, cc]:
                total = gF[cr, cc] + gB[cr, cc]
                if total < best_cost:
                    best_cost = total
                    best_meet = (cr, cc)

            for dr, dc in self.neigh:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if closed_this[nr, nc]:
                    continue
                c_here = sub[nr, nc]
                if not np.isfinite(c_here):
                    continue

                move_cost = math.hypot(dr, dc)
                tentative_g = g_this[current] + c_here + move_cost
                if tentative_g < g_this[nr, nc]:
                    g_this[nr, nc] = tentative_g
                    came_this[(nr, nc)] = current
                    h = self._heur((nr, nc), target)
                    f = tentative_g + self.w * h
                    f_this[nr, nc] = f
                    heapq.heappush(openQ, (f, push_id, (nr, nc)))
                    push_id += 1

                    if g_other[nr, nc] < inf:
                        total = g_this[nr, nc] + g_other[nr, nc]
                        if total < best_cost:
                            best_cost = total
                            best_meet = (nr, nc)

            return push_id, True

        while (openF or openB):
            if (time.monotonic() - t0) > timeout_sec or expansions >= max_expansions:
                rospy.logwarn_throttle(1.0,
                    f"BiA*: early stop (time/exp). exp={expansions}, openF={len(openF)}, openB={len(openB)}")
                break

            expand_forward = False
            if openF and openB:
                expand_forward = openF[0][0] <= openB[0][0]
            elif openF:
                expand_forward = True
            else:
                expand_forward = False

            if expand_forward:
                push_idF, _ = expand(openF, gF, fF, gB, closedF, cameF, g, push_idF, forward=True)
            else:
                push_idB, _ = expand(openB, gB, fB, gF, closedB, cameB, s, push_idB, forward=False)

            if best_meet is not None:
                lb = inf
                if openF: lb = min(lb, openF[0][0])
                if openB: lb = min(lb, openB[0][0])
                if lb >= best_cost:
                    break

        if best_meet is None:
            if gF[g] < inf:
                path_roi = self._reconstruct(cameF, g, s, g, None)
            elif gB[s] < inf:
                path_roi = self._reconstruct(cameB, s, g, s, None)
            else:
                if np.isfinite(fF).any():
                    best_idx = np.unravel_index(np.nanargmin(fF), fF.shape)
                    if np.isfinite(gF[best_idx]):
                        path_roi = self._reconstruct(cameF, best_idx, s, g, None)
                    else:
                        return None
                else:
                    return None
        else:
            path_roi = self._reconstruct(cameF, best_meet, s, g, cameB)

        path_full = [(r0 + r, c0 + c) for (r, c) in path_roi]
        return path_full
      
class AStarPlanner:
    def __init__(self, heuristic_weight=1.15):
        # 8-connectivity
        self.neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        self.w = float(heuristic_weight)

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def plan(self, cost_map, start, goal, timeout_sec=0.5, max_expansions=80000):
        start = tuple(start)
        goal  = tuple(goal)
        rows, cols = cost_map.shape

        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            rospy.logwarn(f"A*: Start {start} out of bounds.")
            return None
        if not (0 <= goal[0]  < rows and 0 <= goal[1]  < cols):
            rospy.logwarn(f"A*: Goal  {goal}  out of bounds.")
            return None
        if not np.isfinite(cost_map[start]):
            rospy.logwarn("A*: Start is an obstacle.")
            return None
        if not np.isfinite(cost_map[goal]):
            rospy.logwarn("A*: Goal is an obstacle.")
            return None

        t0 = time.monotonic()

        # Open set: (f, idx, node)
        open_set = []
        heapq.heappush(open_set, (0.0, 0, start))

        came_from = {}
        g_score = np.full(cost_map.shape, np.inf, dtype=np.float32)
        f_score = np.full(cost_map.shape, np.inf, dtype=np.float32)
        closed  = np.zeros(cost_map.shape, dtype=bool)

        g_score[start] = 0.0
        f_score[start] = self.w * self.heuristic(start, goal)

        push_idx = 1

        best_node = start
        best_f    = f_score[start]

        expansions = 0

        while open_set:
            if (time.monotonic() - t0) > timeout_sec or expansions >= max_expansions:
                rospy.logwarn_throttle(1.0,
                    f"A*: Stopping early (time or expansions). "
                    f"exp={expansions}, open={len(open_set)}")
                if best_node != start:
                    return self._reconstruct_path(came_from, best_node)
                return None

            _, _, current = heapq.heappop(open_set)

            if closed[current]:
                continue
            closed[current] = True
            expansions += 1

            if current == goal:
                return self._reconstruct_path(came_from, current)

            cr, cc = current
            for dr, dc in self.neighbors:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if closed[nr, nc]:
                    continue

                neighbor_cost = cost_map[nr, nc]
                if not np.isfinite(neighbor_cost):
                    continue

                move_cost = math.hypot(dr, dc)
                tentative_g = g_score[current] + neighbor_cost + move_cost

                if tentative_g < g_score[nr, nc]:
                    came_from[(nr, nc)] = current
                    g_score[nr, nc] = tentative_g
                    f = tentative_g + self.w * self.heuristic((nr, nc), goal)
                    f_score[nr, nc] = f
                    heapq.heappush(open_set, (f, push_idx, (nr, nc)))
                    push_idx += 1

                    if f < best_f:
                        best_f = f
                        best_node = (nr, nc)

        rospy.logwarn("A*: Failed to find a path (open set exhausted).")
        return None

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


class RobotCentricPlanner:
    def __init__(self):
        rospy.init_node('robot_centric_planner')

        self.current_odom = None
        self.cost_map = None
        self.map_geom = {}  # {'res','rows','cols'}
        self.map_frame = "ego_vehicle"  # robot-centered grid frame
        self.world_frame = "map"         # path output frame

        self.clamp_to_local_map = True  # clamp long goals to the reachable radius
        self.edge_margin_cells = 2

        # self.planner = AStarPlanner()
        self.planner = BiAStarPlanner(heuristic_weight=1.15, use_8conn=True)


        rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/elevation_mapping1/elevation_map_raw',
                         GridMapMsg, self.map_callback)
        rospy.Subscriber('/move_base_simple/goal',
                         PoseStamped, self.goal_callback)

        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1, latch=True)
        rospy.loginfo("Robot-Centric Planner initialized (Manual transforms).")

    # ----------------- Callbacks -----------------

    def odom_callback(self, msg: Odometry):
        self.current_odom = msg
        self.world_frame = msg.header.frame_id

    def map_callback(self, msg: GridMapMsg):
        self.map_frame = msg.info.header.frame_id or self.map_frame

        layer_name = 'traversability2'
        try:
            layer_index = msg.layers.index(layer_name)
        except ValueError:
            rospy.logwarn_throttle(5.0, f"Layer '{layer_name}' not in GridMap.")
            self.cost_map = None
            return

        layer_data_msg = msg.data[layer_index]
        layout = layer_data_msg.layout

        cols = layout.dim[0].size
        rows = layout.dim[1].size

        data_1d = np.array(layer_data_msg.data, dtype=np.float32)
        trav_map_col_major = data_1d.reshape((cols, rows), order='F')
        trav_map_numpy = trav_map_col_major.T

        cost_map = (1.0 - trav_map_numpy) * 100.0
        self.cost_map = np.nan_to_num(cost_map, nan=np.inf)

        self.map_geom = {
            'res': msg.info.resolution,
            'rows': rows,
            'cols': cols,
        }

    def goal_callback(self, msg: PoseStamped):
        if self.current_odom is None:
            rospy.logwarn("No odometry received. Cannot plan.")
            return
        if self.cost_map is None or not self.map_geom:
            rospy.logwarn("No valid map or geometry received. Cannot plan.")
            return

        rx = self.current_odom.pose.pose.position.x
        ry = self.current_odom.pose.pose.position.y
        q = self.current_odom.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        gx = msg.pose.position.x
        gy = msg.pose.position.y

        if msg.header.frame_id and self.world_frame and msg.header.frame_id != self.world_frame:
            rospy.logwarn(
                f"Goal frame '{msg.header.frame_id}' != odom/world frame '{self.world_frame}'. "
                f"Proceeding as if they are coincident (no TF used)."
            )

        rospy.loginfo(f"Received goal in '{msg.header.frame_id}' frame.")
        rospy.loginfo(f"Current robot pose (world): x={rx:.2f}, y={ry:.2f}, yaw={math.degrees(yaw):.1f}°")
        rospy.loginfo(f"Goal (world): x={gx:.2f}, y={gy:.2f}")

        dx_w = gx - rx
        dy_w = gy - ry
        goal_x_body, goal_y_body = world_to_body(dx_w, dy_w, yaw)

        rospy.loginfo(f"Goal in ego frame: x={goal_x_body:.2f} m (forward), y={goal_y_body:.2f} m (left)")

        geom = self.map_geom
        res = geom['res']
        rows = geom['rows']
        cols = geom['cols']

        dx_cells = -goal_x_body / res
        dy_cells = -goal_y_body / res

        start_r = rows // 2
        start_c = cols // 2
        start_idx = (start_r, start_c)

        if self.clamp_to_local_map:
            max_r = min(start_r, rows - 1 - start_r) - self.edge_margin_cells
            max_c = min(start_c, cols - 1 - start_c) - self.edge_margin_cells
            max_radius = max(0, min(max_r, max_c))
            dist_cells = math.hypot(dx_cells, dy_cells)
            if dist_cells > max_radius and max_radius > 0:
                scale = max_radius / max(1e-6, dist_cells)
                dx_cells *= scale
                dy_cells *= scale
                rospy.logwarn(
                    f"Goal outside local map. Clamped to ~{max_radius * res:.1f} m along the same direction."
                )

        goal_r = int(round(start_r + dx_cells))
        goal_c = int(round(start_c + dy_cells))

        rospy.loginfo(f"Goal offset (cells): rows={int(round(dx_cells))}, cols={int(round(dy_cells))}")

        if not (0 <= goal_r < rows and 0 <= goal_c < cols):
            rospy.logwarn(f"Goal index ({goal_r}, {goal_c}) is outside map bounds ({rows}x{cols}).")
            self.publish_empty_path()
            return

        goal_idx = (goal_r, goal_c)
        rospy.loginfo(f"Planning from grid index {start_idx} to {goal_idx} (NumPy indexing)")

        # path_indices = self.planner.plan(self.cost_map, start_idx, goal_idx)
        path_indices = self.planner.plan(
            self.cost_map,
            start_idx,
            goal_idx,
            timeout_sec=3.57,        # keep callback responsive
            max_expansions=360000,    # safety cap
            roi_margin_cells=400      # tune: larger if many detours required
        )
        
        if path_indices is None:
            rospy.logwarn("No path found.")
            self.publish_empty_path()
            return

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.world_frame

        center_r = rows // 2
        center_c = cols // 2

        for (r, c) in path_indices:
            row_off = center_r - r
            col_off = center_c - c
            pos_x_ego = row_off * res    # +forward
            pos_y_ego = col_off * res    # +left

            dx_w, dy_w = body_to_world(pos_x_ego, pos_y_ego, yaw)
            wx = rx + dx_w
            wy = ry + dy_w

            pose_world = PoseStamped()
            pose_world.header.stamp = path_msg.header.stamp
            pose_world.header.frame_id = self.world_frame
            pose_world.pose.position.x = wx
            pose_world.pose.position.y = wy
            pose_world.pose.orientation.w = 1.0  # no heading for path points
            path_msg.poses.append(pose_world)

        self.path_pub.publish(path_msg)
        rospy.loginfo(f"Path with {len(path_msg.poses)} poses published.")

    def publish_empty_path(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.world_frame
        self.path_pub.publish(path_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        planner_node = RobotCentricPlanner()
        planner_node.run()
    except rospy.ROSInterruptException:
        pass
