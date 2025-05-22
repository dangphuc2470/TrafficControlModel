import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
        
        # Vehicle type distribution (matching the N-S Dominant preset)
        self.vehicle_types = {
            "veh_passenger": 45,
            "veh_bus": 10,
            "veh_truck": 2,
            "veh_emergency": 3,
            "veh_motorcycle": 40
        }

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """ 
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="veh_passenger" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" guiShape="passenger" />
            <vType accel="0.8" decel="3.0" id="veh_bus" length="12.0" minGap="3.0" maxSpeed="20" sigma="0.5" guiShape="bus" />
            <vType accel="0.7" decel="2.5" id="veh_truck" length="10.0" minGap="3.0" maxSpeed="18" sigma="0.5" guiShape="truck" />
            <vType accel="1.2" decel="5.0" id="veh_emergency" length="6.0" minGap="2.0" maxSpeed="30" sigma="0.5" guiShape="emergency" />
            <vType accel="1.5" decel="6.0" id="veh_motorcycle" length="2.0" minGap="1.5" maxSpeed="35" sigma="0.5" guiShape="motorcycle" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                # Select vehicle type based on distribution
                vehicle_type = np.random.choice(
                    list(self.vehicle_types.keys()),
                    p=[v/100 for v in self.vehicle_types.values()]
                )
                
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="%s" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="%s" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="%s" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="%s" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="%s" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="%s" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="%s" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="%s" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="%s" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="%s" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="%s" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="%s" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, vehicle_type, step), file=routes)

            print("</routes>", file=routes)
