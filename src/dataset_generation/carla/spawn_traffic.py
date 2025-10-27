#!/usr/bin/env python3
# spawn_traffic.py — CARLA 0.9.13
# Spawns vehicles using Traffic Manager and pedestrians with AI controllers.

import argparse
import random
import sys
import time
from collections import defaultdict

try:
    import carla
except ImportError:
    raise SystemExit("Cannot import CARLA. Make sure the Python egg for 0.9.13 is in PYTHONPATH.")

def pick_vehicle_blueprints(blueprints, safe=True):
    """Filter vehicle blueprints; optionally avoid very slow or fragile ones."""
    veh_bps = blueprints.filter('vehicle.*')
    if not safe:
        return veh_bps
    # Exclude trailers and bikes for more robust autopilot traffic
    safe_ids = [bp.id for bp in veh_bps if not (
        'microlino' in bp.id or
        't2' in bp.id or
        'carlacola' in bp.id or
        'isetta' in bp.id or
        'harley' in bp.id or
        'kawasaki' in bp.id or
        'yamaha' in bp.id or
        'omni' in bp.id or
        'firetruck' in bp.id or
        'ambulance' in bp.id or
        'police' in bp.id or
        'bike' in bp.id or
        'bh' in bp.id
    )]
    return [bp for bp in veh_bps if bp.id in safe_ids]

def set_random_vehicle_attributes(bp):
    if bp.has_attribute('color'):
        bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
    if bp.has_attribute('driver_id'):
        bp.set_attribute('driver_id', random.choice(bp.get_attribute('driver_id').recommended_values))
    if bp.has_attribute('is_invincible'):
        bp.set_attribute('is_invincible', 'true')
    # Keep lights on
    if bp.has_attribute('lights'):
        bp.set_attribute('lights', 'True')

def make_walker_bp(blueprints):
    wbp = random.choice(blueprints.filter('walker.pedestrian.*'))

    # Keep pedestrians mortal so collisions register normally
    if wbp.has_attribute('is_invincible'):
        wbp.set_attribute('is_invincible', 'false')

    # Set a walking/running speed using the blueprint's recommended values
    speed_attr = wbp.get_attribute('speed') if wbp.has_attribute('speed') else None
    if speed_attr is not None:
        speeds = list(speed_attr.recommended_values)
        if speeds:
            # 70% normal (first recommended value), 30% faster (use the next value if available)
            if len(speeds) == 1:
                chosen = speeds[0]
            else:
                # bias toward the slower option
                chosen = random.choices(
                    [speeds[0], speeds[min(1, len(speeds)-1)]],
                    weights=[7, 3],
                )[0]
            wbp.set_attribute('speed', chosen)

    return wbp


def main():
    parser = argparse.ArgumentParser(description="Spawn vehicles and pedestrians in CARLA 0.9.13.")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8000, help='Traffic Manager port')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vehicles', type=int, default=57)
    parser.add_argument('--walkers', type=int, default=40)
    parser.add_argument('--safe-vehicles', action='store_true', help='Stricter vehicle model filter')
    parser.add_argument('--no-sync', action='store_true', help='Do not force synchronous mode')
    parser.add_argument('--tm-ignore-lights', action='store_true', help='Let TM ignore traffic lights')
    parser.add_argument('--tm-perc-speed', type=float, default=100.0, help='TM global speed percentage (100=default)')
    parser.add_argument('--tm-hybrid', action='store_true', help='Enable TM hybrid physics mode for large crowds')
    args = parser.parse_args()

    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    original_settings = world.get_settings()

    # Traffic Manager
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(not args.no_sync)
    tm.global_percentage_speed_difference(100.0 - args.tm_perc_speed)  # 0 = default, positive = slower
    tm.set_random_device_seed(args.seed)
    if args.tm_ignore_lights:
        tm.set_ignore_traffic_lights(True)
    if args.tm_hybrid:
        tm.set_hybrid_physics_mode(True)
        tm.set_hybrid_physics_radius(70.0)

    # Enable synchronous simulation for determinism and stable spawns
    if not args.no_sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS sim step
        settings.substepping = True
        settings.max_substeps = 10
        world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = pick_vehicle_blueprints(blueprint_library, safe=True)
    walker_controller_bp = blueprint_library.find('controller.ai.walker')

    # --- Spawn Vehicles ---
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    num_vehicles = min(args.vehicles, len(spawn_points))

    vehicle_batch = []
    for i in range(num_vehicles):
        bp = random.choice(vehicle_blueprints)
        set_random_vehicle_attributes(bp)
        tr = spawn_points[i]
        vehicle_batch.append(carla.command.SpawnActor(bp, tr))

    vehicles_list = []
    if vehicle_batch:
        responses = client.apply_batch_sync(vehicle_batch, args.no_sync is True)  # blocking in sync mode
        for r in responses:
            if r.error:
                continue
            vehicles_list.append(r.actor_id)

    # Put vehicles under Traffic Manager
    if vehicles_list:
        autopilot_batch = [carla.command.SetAutopilot(v_id, True, args.tm_port) for v_id in vehicles_list]
        client.apply_batch_sync(autopilot_batch, True)

        # Per-vehicle options (lane changes, lights, etc.)
        for v_id in vehicles_list:
            tm.distance_to_leading_vehicle(world.get_actor(v_id), 2.5)
            tm.auto_lane_change(world.get_actor(v_id), True)

    # --- Spawn Walkers (pedestrians) ---
    walkers_to_spawn = args.walkers
    walker_spawns = []
    for _ in range(walkers_to_spawn):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            yaw = random.uniform(-180.0, 180.0)
            walker_spawns.append(carla.Transform(loc, carla.Rotation(yaw=yaw)))

    walker_batch = []
    walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
    for tr in walker_spawns:
        wbp = make_walker_bp(blueprint_library)
        walker_batch.append(carla.command.SpawnActor(wbp, tr))

    walkers_list = []
    if walker_batch:
        responses = client.apply_batch_sync(walker_batch, True)
        for r in responses:
            if r.error:
                continue
            walkers_list.append(r.actor_id)

    # Spawn controllers for walkers
    controller_batch = []
    for wid in walkers_list:
        controller_batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), wid))

    controllers_list = []
    if controller_batch:
        responses = client.apply_batch_sync(controller_batch, True)
        for r in responses:
            if r.error:
                continue
            controllers_list.append(r.actor_id)

    world.tick() if not args.no_sync else None

    # Start controllers and send them walking
    all_actors = world.get_actors(walkers_list + controllers_list)
    walker_speeds = defaultdict(lambda: 1.4)  # default max speed
    for ctrl_id in controllers_list:
        ctrl = world.get_actor(ctrl_id)
        ctrl.start()
        # Random destination on the navmesh
        dest = world.get_random_location_from_navigation()
        if dest:
            ctrl.go_to_location(dest)
        # Random speed factor: normal vs jogging
        wid = ctrl.parent.id
        speed = 1.3 if random.random() < 0.7 else 2.0
        walker_speeds[wid] = speed
        ctrl.set_max_speed(speed)

    print(f"Spawned {len(vehicles_list)} vehicles and {len(walkers_list)} walkers.")
    print("Press Ctrl+C to clean up and restore settings.")

    try:
        if not args.no_sync:
            # Keep sim ticking deterministically
            while True:
                world.tick()
        else:
            # Asynchronous fallback
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        print("\nCleaning up actors...")
        # Stop walker controllers first
        for cid in controllers_list:
            ctrl = world.get_actor(cid)
            if ctrl is not None:
                try:
                    ctrl.stop()
                except RuntimeError:
                    pass

        to_destroy = controllers_list + walkers_list + vehicles_list
        if to_destroy:
            client.apply_batch([carla.command.DestroyActor(x) for x in to_destroy])

        # Restore settings
        if not args.no_sync:
            world.apply_settings(original_settings)
        # Return TM to async if we changed it
        tm.set_synchronous_mode(False)
        print("Done.")

if __name__ == '__main__':
    main()
