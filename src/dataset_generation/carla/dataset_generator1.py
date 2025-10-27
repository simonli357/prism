#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import time
import logging
import os
import sys
import csv
import json
import signal
import yaml
import errno
import threading

# ---------- CARLA ----------
import carla
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

# ---------- ROS ----------
import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge
import cv2
import math, logging

def speed_mps(v):
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def setup_logging(log_file: str = "", verbose: bool = True):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    logging.info("Logging initialized%s",
                 f" (also to {log_file})" if log_file else "")

def fmt_dur(seconds: float) -> str:
    if seconds < 0: return "n/a"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# =========================
# MOVER (CARLA control)
# =========================
def find_ego(world, role_name="ego_vehicle"):
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name") == role_name:
            return a
    return None

def wait_for_ego(world, role_name="ego_vehicle", timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ego = find_ego(world, role_name)
        if ego:
            return ego
        world.wait_for_tick()
    raise RuntimeError("ego_vehicle not found. Ensure your ROS launch has spawned it in this map.")

def set_sync(world, fps):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / float(fps)
    world.apply_settings(settings)

def setup_tm(client, tm_port, hybrid_physics=False, seed=42):
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.0)
    tm.set_random_device_seed(seed)
    try:
        tm.set_hybrid_physics_mode(hybrid_physics)
    except Exception:
        pass
    try:
        tm.set_respawn_dormant_vehicles(True)
    except Exception:
        pass
    return tm

def set_random_weather(world, night_prob=0.25):
    sun = random.uniform(-15, 75) if random.random() > night_prob else random.uniform(-20, 5)
    w = carla.WeatherParameters(
        cloudiness=random.uniform(0, 90),
        precipitation=random.uniform(0, 60),
        precipitation_deposits=random.uniform(0, 50),
        wind_intensity=random.uniform(0, 50),
        sun_azimuth_angle=random.uniform(0, 360),
        sun_altitude_angle=sun,
        fog_density=random.uniform(0, 40),
        fog_distance=random.uniform(0, 80),
        wetness=random.uniform(0, 60),
    )
    world.set_weather(w)

def random_destination(world):
    spawns = world.get_map().get_spawn_points()
    tf = random.choice(spawns)
    return carla.Location(tf.location.x, tf.location.y, tf.location.z)

def spawn_traffic(world, client, tm, n_vehicles=100, n_walkers=50, safe=True):
    bp_lib = world.get_blueprint_library()

    vehicles = []
    if n_vehicles > 0:
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        spawn_points = spawn_points[:n_vehicles]

        batch = []
        for sp in spawn_points:
            bp = random.choice(bp_lib.filter("vehicle.*"))
            if safe and bp.has_attribute("number_of_wheels"):
                try:
                    if int(bp.get_attribute("number_of_wheels")) < 4:
                        continue
                except Exception:
                    pass
            batch.append(carla.command.SpawnActor(bp, sp))

        results = client.apply_batch_sync(batch, True)
        for r in results:
            if r.error:
                continue
            v = world.get_actor(r.actor_id)
            if v is None:
                continue
            try:
                v.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(v, random.uniform(-10, 30))
            except Exception:
                pass
            vehicles.append(v)

    # Walkers
    walkers = []
    walker_controllers = []
    if n_walkers > 0:
        controller_bp = bp_lib.find("controller.ai.walker")

        walker_bps = bp_lib.filter("walker.pedestrian.*")
        spawn_transforms = []
        for _ in range(n_walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                spawn_transforms.append(carla.Transform(loc))

        walker_batch = []
        for tf in spawn_transforms:
            wbp = random.choice(walker_bps)
            walker_batch.append(carla.command.SpawnActor(wbp, tf))

        walker_results = client.apply_batch_sync(walker_batch, True)
        walker_ids = [r.actor_id for r in walker_results if not r.error]

        ctrl_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
        ctrl_results = client.apply_batch_sync(ctrl_batch, True)
        ctrl_ids = [r.actor_id for r in ctrl_results if not r.error]

        for wid, cid in zip(walker_ids, ctrl_ids):
            w = world.get_actor(wid)
            c = world.get_actor(cid)
            if w and c:
                walkers.append(w)
                walker_controllers.append(c)
                try:
                    c.start()
                    c.go_to_location(world.get_random_location_from_navigation())
                    c.set_max_speed(random.uniform(0.5, 1.5))
                except Exception:
                    pass

    return vehicles, walkers, walker_controllers

def cleanup(world, client, vehicles, walkers, walker_controllers):
    for c in walker_controllers:
        try:
            c.stop()
        except Exception:
            pass
    actors = [a for a in (vehicles + walkers + walker_controllers) if a is not None]
    if actors:
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])

def set_agent_speed(agent, kph):
    if hasattr(agent, "set_target_speed"):
        agent.set_target_speed(kph)
        return
    if hasattr(agent, "_local_planner") and hasattr(agent._local_planner, "set_speed"):
        agent._local_planner.set_speed(kph)

def mover_thread_fn(args, stop_event: threading.Event):
    logger = logging.getLogger()
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)

    # Use current world to keep ROS-spawned ego/sensors
    world = client.get_world()
    town = world.get_map().name
    logger.info("[roam] Using current world: %s", town)

    set_sync(world, args.fps)
    tm = setup_tm(client, tm_port=args.tm_port, hybrid_physics=args.hybrid_physics)

    ego = wait_for_ego(world, role_name=args.role_name, timeout=30.0)
    agent = BehaviorAgent(ego, behavior="normal") if (not args.basic_agent) else BasicAgent(ego, target_speed=30.0)
    agent.set_destination(random_destination(world))
    set_agent_speed(agent, 30.0)

    vehicles = []; walkers_list = []; walker_ctrls = []
    if args.traffic > 0 or args.walkers > 0:
        logger.info("[roam] Spawning traffic: vehicles=%d, walkers=%d", args.traffic, args.walkers)
        vehicles, walkers_list, walker_ctrls = spawn_traffic(world, client, tm, args.traffic, args.walkers)

    # Progress bookkeeping
    sim_ticks_total = int(args.minutes_per_town * 60 * args.fps) if args.minutes_per_town > 0 else -1
    tick_count = 0
    start_wall = time.time()
    last_log_t = start_wall
    last_weather_t = start_wall

    logger.info("[roam] Driving for %s at %d FPS (%s ticks)...",
                ("∞" if args.minutes_per_town <= 0 else f"{args.minutes_per_town} sim minutes"),
                args.fps,
                ("unbounded" if sim_ticks_total < 0 else f"target {sim_ticks_total}"))

    try:
        last_loc = ego.get_location()
        last_progress_tick = 0
        while not stop_event.is_set():
            now = time.time()
            if now - last_log_t >= args.log_interval_seconds:
                wall_elapsed = now - start_wall
                wall_fps = (tick_count / wall_elapsed) if wall_elapsed > 0 else 0.0
                sim_minutes = tick_count / (args.fps * 60.0)
                elapsed_str = fmt_dur(wall_elapsed)
                logger.info("[time] Elapsed wall time: %s (sim=%.2f min)", elapsed_str, sim_minutes)

            if now - last_weather_t >= args.weather_interval_seconds:
                set_random_weather(world)
                last_weather_t = now

            done = False
            if hasattr(agent, "done"):
                try:
                    done = agent.done()
                except Exception:
                    done = False
            if done:
                agent.set_destination(random_destination(world))

            control = agent.run_step()
            ego.apply_control(control)

            world.tick()
            tick_count += 1

            spd = speed_mps(ego.get_velocity())
            loc = ego.get_location()
            moved = loc.distance(last_loc)    # CARLA Location has distance()

            if moved > 1.5 or spd > 0.2:
                last_loc = loc
                last_progress_tick = tick_count

            stalled_sec = (tick_count - last_progress_tick) / float(args.fps)
            if stalled_sec > 8.0:
                logging.warning("[unstick] %.1fs without progress; new destination.", stalled_sec)
                ego.apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
                agent.set_destination(random_destination(world))
                last_loc = ego.get_location()
                last_progress_tick = tick_count

            # Log progress every N seconds (wall)
            if now - last_log_t >= args.log_interval_seconds:
                wall_elapsed = now - start_wall
                wall_fps = (tick_count / wall_elapsed) if wall_elapsed > 0 else 0.0
                sim_minutes = tick_count / (args.fps * 60.0)
                if sim_ticks_total > 0:
                    pct = (100.0 * tick_count) / sim_ticks_total
                    ticks_left = max(0, sim_ticks_total - tick_count)
                    eta_sec = (ticks_left / wall_fps) if wall_fps > 0 else -1
                    logger.info("[progress] ticks=%d/%d (%.1f%%) | sim=%.2f min | wall=%s | wall_fps=%.2f | ETA=%s",
                                tick_count, sim_ticks_total, pct, sim_minutes, fmt_dur(wall_elapsed),
                                wall_fps, (fmt_dur(eta_sec) if eta_sec >= 0 else "n/a"))
                else:
                    logger.info("[progress] ticks=%d | sim=%.2f min | wall=%s | wall_fps=%.2f",
                                tick_count, sim_minutes, fmt_dur(wall_elapsed), wall_fps)
                last_log_t = now

            # Stop on sim-time limit if set
            if sim_ticks_total > 0 and tick_count >= sim_ticks_total:
                logger.info("[roam] Reached target ticks; stopping mover.")
                break
    finally:
        cleanup(world, client, vehicles, walkers_list, walker_ctrls)
        wall_total = time.time() - start_wall
        logger.info("[roam] Finished | ticks=%d | sim=%.2f min | wall=%s | avg wall_fps=%.2f",
                    tick_count, tick_count/(args.fps*60.0), fmt_dur(wall_total),
                    (tick_count/wall_total) if wall_total>0 else 0.0)

# =========================
# RECORDER (ROS)
# =========================
def _safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def _get_last_frame_idx(index_csv_path):
    if not os.path.isfile(index_csv_path):
        return -1
    last_idx = -1
    try:
        with open(index_csv_path, "r") as f:
            _ = f.readline()  # header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    comma = line.find(",")
                    if comma == -1:
                        continue
                    fi = int(line[:comma])
                    if fi > last_idx:
                        last_idx = fi
                except Exception:
                    continue
    except Exception:
        pass
    return last_idx

CAMERAS = ["front", "left", "rear", "right"]
DEFAULT_TOPICS = {
    "semantic": {
        "front": "/carla/ego_vehicle/semantic_camera_front/image",
        "left":  "/carla/ego_vehicle/semantic_camera_left/image",
        "rear":  "/carla/ego_vehicle/semantic_camera_rear/image",
        "right": "/carla/ego_vehicle/semantic_camera_right/image",
    },
    "rgb": {
        "front": "/carla/ego_vehicle/camera_front/image",
        "left":  "/carla/ego_vehicle/camera_left/image",
        "rear":  "/carla/ego_vehicle/camera_rear/image",
        "right": "/carla/ego_vehicle/camera_right/image",
    },
}
DEFAULT_INFO_TOPICS = {
    "semantic": {
        "front": "/carla/ego_vehicle/semantic_camera_front/camera_info",
        "left":  "/carla/ego_vehicle/semantic_camera_left/camera_info",
        "rear":  "/carla/ego_vehicle/semantic_camera_rear/camera_info",
        "right": "/carla/ego_vehicle/semantic_camera_right/camera_info",
    },
    "rgb": {
        "front": "/carla/ego_vehicle/camera_front/camera_info",
        "left":  "/carla/ego_vehicle/camera_left/camera_info",
        "rear":  "/carla/ego_vehicle/camera_rear/camera_info",
        "right": "/carla/ego_vehicle/camera_right/camera_info",
    },
}

class CarlaDatasetWriter(object):
    def __init__(self, out_dir,
                 exact_sync=False,
                 slop=0.05,
                 queue_size=30,
                 max_frames=-1):
        self.out_dir = out_dir
        self.frames = 0
        self.max_frames = max_frames
        self.bridge = CvBridge()

        for cam in CAMERAS:
            for sub in ["rgb", "sem"]:
                _safe_makedirs(os.path.join(out_dir, cam, sub))

        self.index_path = os.path.join(out_dir, "index.csv")
        existing_last = _get_last_frame_idx(self.index_path)
        self.frames = existing_last + 1 if existing_last >= 0 else 0

        new_file = not os.path.exists(self.index_path) or existing_last < 0
        self.index_file = open(self.index_path, "a", newline="")
        self.csv = csv.writer(self.index_file)
        if new_file:
            self.csv.writerow([
                "frame_idx","stamp_ns",
                "front_rgb","front_sem",
                "left_rgb","left_sem",
                "rear_rgb","rear_sem",
                "right_rgb","right_sem"
            ])
            self.index_file.flush()
        else:
            rospy.loginfo("Resuming at frame_idx %d", self.frames)

        meta = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exact_sync": exact_sync,
            "slop": slop,
            "queue_size": queue_size,
            "topics": DEFAULT_TOPICS,
            "camera_info_topics": DEFAULT_INFO_TOPICS
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # camera_info
        self._saved_info = set()
        for typ in ["rgb","semantic"]:
            for cam in CAMERAS:
                t = DEFAULT_INFO_TOPICS[typ][cam]
                rospy.Subscriber(t, CameraInfo, self._make_info_cb(cam, typ), queue_size=1)

        # images
        subs_rgb = [message_filters.Subscriber(DEFAULT_TOPICS["rgb"][c], Image) for c in CAMERAS]
        subs_sem = [message_filters.Subscriber(DEFAULT_TOPICS["semantic"][c], Image) for c in CAMERAS]
        all_subs = subs_rgb + subs_sem

        if exact_sync:
            ts = message_filters.TimeSynchronizer(all_subs, queue_size=queue_size)
        else:
            ts = message_filters.ApproximateTimeSynchronizer(all_subs, queue_size=queue_size, slop=slop)
        ts.registerCallback(self.sync_cb)

        rospy.loginfo("Saving to %s | exact_sync=%s slop=%.3f queue=%d max_frames=%d",
                      out_dir, str(exact_sync), slop, queue_size, max_frames)

    def _make_info_cb(self, cam_name, cam_type):
        def cb(msg):
            key = (cam_name, cam_type)
            if key in self._saved_info:
                return
            cam_dir = os.path.join(self.out_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            path = os.path.join(cam_dir, f"camera_info_{cam_type}.yaml")
            if os.path.exists(path):
                self._saved_info.add(key)
                return
            data = {
                "width": msg.width, "height": msg.height,
                "distortion_model": msg.distortion_model,
                "D": list(msg.D), "K": list(msg.K),
                "R": list(msg.R), "P": list(msg.P),
                "binning_x": msg.binning_x, "binning_y": msg.binning_y,
                "roi": {
                    "x_offset": msg.roi.x_offset, "y_offset": msg.roi.y_offset,
                    "height": msg.roi.height, "width": msg.roi.width,
                    "do_rectify": msg.roi.do_rectify,
                },
            }
            with open(path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
            self._saved_info.add(key)
        return cb

    def _save_img(self, cv_img, cam, sub, base):
        rel = os.path.join(cam, sub, base + ".png")
        abspath = os.path.join(self.out_dir, rel)
        cv2.imwrite(abspath, cv_img)
        return rel

    def sync_cb(self,
                rgb_front, rgb_left, rgb_rear, rgb_right,
                sem_front, sem_left, sem_rear, sem_right):
        if rospy.is_shutdown():
            return
        stamp = rgb_front.header.stamp
        stamp_ns = stamp.secs * 1000000000 + stamp.nsecs
        base = f"{self.frames:06d}_{stamp_ns}"

        try:
            img_rgb = {
                "front": self.bridge.imgmsg_to_cv2(rgb_front, desired_encoding="bgr8"),
                "left":  self.bridge.imgmsg_to_cv2(rgb_left,  desired_encoding="bgr8"),
                "rear":  self.bridge.imgmsg_to_cv2(rgb_rear,  desired_encoding="bgr8"),
                "right": self.bridge.imgmsg_to_cv2(rgb_right, desired_encoding="bgr8"),
            }
            img_sem = {
                "front": self.bridge.imgmsg_to_cv2(sem_front),
                "left":  self.bridge.imgmsg_to_cv2(sem_left),
                "rear":  self.bridge.imgmsg_to_cv2(sem_rear),
                "right": self.bridge.imgmsg_to_cv2(sem_right),
            }

            paths = {}
            for cam in CAMERAS:
                paths[(cam,"rgb")] = self._save_img(img_rgb[cam], cam, "rgb", base)
                paths[(cam,"sem")] = self._save_img(img_sem[cam], cam, "sem", base)

            # simple progress echo
            sys.stdout.write(f"\rSaved frame {self.frames} stamp {stamp_ns}")
            sys.stdout.flush()

            self.csv.writerow([
                self.frames, stamp_ns,
                paths[("front","rgb")], paths[("front","sem")],
                paths[("left","rgb")],  paths[("left","sem")],
                paths[("rear","rgb")],  paths[("rear","sem")],
                paths[("right","rgb")], paths[("right","sem")],
            ])
            self.index_file.flush()

            self.frames += 1
            if self.max_frames > 0 and self.frames >= self.max_frames:
                rospy.loginfo("Reached max_frames=%d. Exiting.", self.max_frames)
                rospy.signal_shutdown("done")

        except Exception as e:
            rospy.logerr("Failed to save images for frame %d: %s", self.frames, repr(e))

    def close(self):
        try:
            self.index_file.flush()
            self.index_file.close()
        except Exception:
            pass

# =========================
# CLI + main
# =========================

def build_argparser():
    ap = argparse.ArgumentParser("All-in-one CARLA mover + ROS recorder")
    # mover
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--role-name", default="ego_vehicle")
    ap.add_argument("--minutes_per_town", type=float, default=37.0)  # 0 = run forever
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--traffic", type=int, default=42)
    ap.add_argument("--walkers", type=int, default=42)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--hybrid-physics", action="store_true")
    ap.add_argument("--basic-agent", action="store_true")
    ap.add_argument("--log-interval-seconds", type=float, default=5.0)
    ap.add_argument("--weather-interval-seconds", type=float, default=10.0)
    ap.add_argument("--log-file", default="")
    ap.add_argument("--quiet", action="store_true")
    # recorder
    ap.add_argument("--out", help="Output dataset directory", default="/media/slsecret/T7/town7/")
    ap.add_argument("--exact_sync", action="store_true")
    ap.add_argument("--slop", type=float, default=0.05)
    ap.add_argument("--queue_size", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--no-images", action="store_true", help="Do not subscribe to or save images; only record camera_info and metadata")
    return ap


def main():
    ap = build_argparser()
    # Let ROS strip its remapping args first when parsing recorder args
    # but we need all args here; use sys.argv directly:
    args = ap.parse_args()

    # Recorder init (ROS)
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    rospy.init_node("carla_all_in_one", anonymous=True)

    writer = None
    if not getattr(args, "no_images", False):
        writer_cls = CarlaDatasetWriter
        writer = writer_cls(
            out_dir=args.out,
            exact_sync=args.exact_sync,
            slop=args.slop,
            queue_size=args.queue_size,
            max_frames=args.max_frames
        )

    # Mover in a background thread
    stop_event = threading.Event()
    t = threading.Thread(target=mover_thread_fn, args=(args, stop_event), daemon=True)
    t.start()

    def _shutdown(sig, frame):
        stop_event.set()
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        try:
            rospy.signal_shutdown("signal")
        except Exception:
            pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Spin until ROS shutdown
    rospy.spin()
    try:
        if writer is not None:
            writer.close()
    except Exception:
        pass
    stop_event.set()
    t.join(timeout=5.0)

if __name__ == "__main__":
    main()
