#!/usr/bin/env python3
import carla

def main():
    try:
        # 1. Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()

        # 2. Get all traffic light and sign objects
        # Note the plural 'TrafficSigns'
        traffic_lights = world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
        traffic_signs = world.get_environment_objects(carla.CityObjectLabel.TrafficSigns)

        # 3. Store the IDs of all objects you want to remove in a set
        objects_to_toggle = set()
        for obj in traffic_lights:
            objects_to_toggle.add(obj.id)
        # for obj in traffic_signs:
        #     objects_to_toggle.add(obj.id)

        # 4. Toggle their visibility OFF
        if objects_to_toggle:
            print(f"Disabling visibility for {len(objects_to_toggle)} traffic objects.")
            world.enable_environment_objects(objects_to_toggle, False)
        else:
            print("No traffic lights or signs found to disable.")

        # To toggle them back ON, you would run this:
        # world.enable_environment_objects(objects_to_toggle, True)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()