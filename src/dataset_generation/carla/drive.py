#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
import time
import logging
import math
import sys

import carla
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

def setup_logging(verbose: bool = True):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging initialized.")

def fmt_dur(seconds: float) -> str:
    if seconds < 0: return "n/a"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def cleanup(client, vehicles, walkers, walker_controllers):
    logging.info("Destroying %d vehicles and %d walkers.", len(vehicles), len(walkers))
    for controller in walker_controllers:
        try:
            controller.stop()
        except RuntimeError as e:
            logging.warning(f"Error stopping walker controller: {e}")
            
    actors = [actor for actor in (vehicles + walkers + walker_controllers) if actor is not None and actor.is_alive]
    if actors:
        client.apply_batch([carla.command.DestroyActor(actor) for actor in actors])
    logging.info("Cleanup complete.")

def run_simulation(args):
    client = None
    vehicles, walkers, walker_controllers = [], [], []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        logging.info("Connected to CARLA server on map %s", world.get_map().name)

        # Setup Traffic Manager
        tm = client.get_trafficmanager(args.tm_port)
        # tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)
        # tm.set_hybrid_physics_mode(args.hybrid_physics) # This can be enabled if needed

        # Wait for the ego vehicle, which should be spawned externally
        ego_vehicle = None
        start_time = time.time()
        while ego_vehicle is None:
            logging.info("Waiting for ego vehicle with role_name '%s'...", args.role_name)
            possible_vehicles = world.get_actors().filter("vehicle.*")
            for vehicle in possible_vehicles:
                if vehicle.attributes.get("role_name") == args.role_name:
                    ego_vehicle = vehicle
                    break
            if time.time() > start_time + 20: # 20 second timeout
                raise RuntimeError("Could not find ego vehicle. Please ensure it's spawned in the world.")
            world.wait_for_tick()
        logging.info("Ego vehicle found: %s", ego_vehicle.type_id)

        vehicles, walkers, walker_controllers = spawn_actors(world, client, tm, args.traffic, args.walkers)
        
        # Create the agent to control the ego vehicle
        agent = BehaviorAgent(ego_vehicle, behavior="aggressive")
        # agent = BehaviorAgent(ego_vehicle, behavior="normal")
        agent.set_destination(random.choice(world.get_map().get_spawn_points()).location)

        duration = 10000 if args.duration <= 0 else args.duration
        sim_ticks_total = int(duration * 60 * args.fps)
        tick_count = 0
        start_wall_time = time.time()
        logging.info("Starting simulation for %.2f minutes (%d ticks)...", duration, sim_ticks_total)

        while args.duration < 0 or tick_count < sim_ticks_total:
            if agent.done():
                logging.info("Agent reached destination, setting a new random one.")
                agent.set_destination(random.choice(world.get_map().get_spawn_points()).location)

            control = agent.run_step()
            ego_vehicle.apply_control(control)

            tick_count += 1
            
            if tick_count % (args.fps * 10) == 0: # Log every 10 seconds
                elapsed_time = time.time() - start_wall_time
                percent_done = (tick_count / sim_ticks_total) * 100
                logging.info("Progress: %.1f%% | Ticks: %d/%d | Wall time: %s", 
                             percent_done, tick_count, sim_ticks_total, fmt_dur(elapsed_time))
        
        logging.info("Simulation duration reached.")

    except RuntimeError as e:
        logging.error("A runtime error occurred: %s", e)
    except KeyboardInterrupt:
        logging.info("\nSimulation interrupted by user.")
    finally:
        if client:
            cleanup(client, vehicles, walkers, walker_controllers)

def spawn_actors(world, client, tm, n_vehicles, n_walkers):
    bp_lib = world.get_blueprint_library()
    vehicles = []
    walkers = []
    walker_controllers = []

    # Spawn Vehicles
    if n_vehicles > 0:
        logging.info("Spawning %d vehicles...", n_vehicles)
        spawn_points = world.get_map().get_spawn_points()
        if n_vehicles > len(spawn_points):
            logging.warning("%d vehicles requested, but map only has %d spawn points. Capping.", n_vehicles, len(spawn_points))
            n_vehicles = len(spawn_points)
        random.shuffle(spawn_points)
        
        vehicle_bps = bp_lib.filter("vehicle.*")
        vehicle_bps = [
            bp for bp in vehicle_bps 
            if 'truck' not in bp.id 
            and 'van' not in bp.id
        ]

        batch = [carla.command.SpawnActor(random.choice(vehicle_bps), sp).then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())) for sp in spawn_points[:n_vehicles]]
        
        results = client.apply_batch_sync(batch, True)
        vehicles = [world.get_actor(res.actor_id) for res in results if not res.error]
        logging.info("Successfully spawned %d vehicles.", len(vehicles))

    # Spawn Walkers
    if n_walkers > 0:
        logging.info("Spawning %d walkers...", n_walkers)
        try:
            walker_bp = bp_lib.filter("walker.pedestrian.*")
            controller_bp = bp_lib.find("controller.ai.walker")

            # Spawn walkers
            spawn_locations = [world.get_random_location_from_navigation() for _ in range(n_walkers)]
            batch_walkers = [carla.command.SpawnActor(random.choice(walker_bp), carla.Transform(loc)) for loc in spawn_locations if loc]
            results_walkers = client.apply_batch_sync(batch_walkers, True)
            walker_ids = [res.actor_id for res in results_walkers if not res.error]
            walkers = [world.get_actor(wid) for wid in walker_ids]
        except Exception as e:
            logging.error("Error spawning walkers: %s", e)
            # delete any spawned vehicles before exiting
            cleanup(client, vehicles, walkers, walker_controllers)
            raise e
        print("spawned {} walkers".format(len(walkers)))
        
        # Spawn walker controllers
        batch_controllers = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
        results_controllers = client.apply_batch_sync(batch_controllers, True)
        controller_ids = [res.actor_id for res in results_controllers if not res.error]
        walker_controllers = [world.get_actor(cid) for cid in controller_ids]
        
        print("successfully spawned {} walkers and {} controllers".format(len(walkers), len(walker_controllers)))
        # Start walkers
        world.wait_for_tick() # Wait for controllers to be assigned
        for controller in walker_controllers:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.0 + random.random()) # Speed between 1.0 and 2.0 m/s
        
        logging.info("Successfully spawned %d walkers.", len(walkers))

    return vehicles, walkers, walker_controllers

def main():
    ap = argparse.ArgumentParser(description="CARLA script to spawn traffic and drive an ego vehicle randomly.")
    ap.add_argument("--host", default="localhost", help="CARLA server host")
    ap.add_argument("--port", type=int, default=2000, help="CARLA server port")
    ap.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    ap.add_argument("--role-name", default="ego_vehicle", help="Role name of the vehicle to control")
    ap.add_argument("--duration", type=float, default=-1.0, help="Duration of the simulation in minutes")
    ap.add_argument("--fps", type=int, default=20, help="Simulation FPS")
    ap.add_argument("--traffic", type=int, default=37, help="Number of vehicles to spawn")
    ap.add_argument("--walkers", type=int, default=0, help="Number of walkers to spawn")
    ap.add_argument("--quiet", action="store_true", help="Disable verbose logging")
    
    args = ap.parse_args()
    
    setup_logging(verbose=not args.quiet)
    run_simulation(args)

if __name__ == "__main__":
    main()