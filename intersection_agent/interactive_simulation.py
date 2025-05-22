import traci
import numpy as np
import timeit
import os
import time
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox, QSplitter
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from agent_communicator import AgentCommunicatorTesting
from main_window import MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

class InteractiveSimulation(QObject):
    # Define signals
    step_updated = pyqtSignal(int)
    vehicle_updated = pyqtSignal(dict)
    stats_updated = pyqtSignal(dict)
    cumulative_stats_updated = pyqtSignal(dict)

    def __init__(self, Model, sumo_cmd, max_steps, green_duration, yellow_duration,
                 num_states, num_actions, server_url=None, agent_id=None,
                 mapping_config=None, env_file_path=None):
        # Initialize QObject
        super().__init__()

        self._Model = Model
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []
        self._vehicle_counter = 0

        # Add vehicle tracking
        self._active_vehicles = set()  # Track vehicles currently in simulation
        self._exited_vehicles = {}  # Track vehicles that have exited and their destinations
        self._incoming_vehicles = []  # Track vehicles that should be spawned

        # Define phase durations (in seconds)
        self.phase_durations = {
            0: 31,  # NS Green
            1: 2,   # NS Yellow
            2: 15,  # NSL Green
            3: 2,   # NSL Yellow
            4: 31,  # EW Green
            5: 2,   # EW Yellow
            6: 15,  # EWL Green
            7: 2    # EWL Yellow
        }

        # Simulation control
        self.running = False
        self.step = 0
        self.speed = 2.0
        self.auto_spawn = True
        self.spawn_interval = 4
        self.spawn_interval_random = True
        self.min_interval = 2
        self.max_interval = 6
        self.spawn_count = 5
        self.spawn_count_random = True
        self.min_count = 1
        self.max_count = 8
        self.last_spawn_step = 0

        # Road IDs and their connections
        self.roads = {
            'north': 'N2TL',
            'south': 'S2TL',
            'east': 'E2TL',
            'west': 'W2TL'
        }

        # Map road IDs to connected intersections
        self.road_connections = {
            'N2TL': 'agent2',  # North road connects to agent2
            'S2TL': 'agent3',  # South road connects to agent3
            'E2TL': 'agent4',  # East road connects to agent4
            'W2TL': 'agent1'   # West road connects to agent1
        }

        # Vehicle type distribution (N-S Dominant preset)
        self.vehicle_types = {
            "veh_passenger": 45,
            "veh_bus": 10,
            "veh_truck": 2,
            "veh_emergency": 3,
            "veh_motorcycle": 40
        }

        # Speed ranges for different vehicle types (reduced to prevent too high speeds)
        self.speed_ranges = {
            "veh_passenger": (3, 8),    # Reduced from (5, 15)
            "veh_bus": (2, 6),          # Reduced from (3, 10)
            "veh_truck": (2, 5),        # Reduced from (3, 8)
            "veh_emergency": (4, 10),   # Reduced from (8, 20)
            "veh_motorcycle": (3, 8)    # Reduced from (7, 18)
        }

        # Route distribution (N-S Dominant preset)
        self.route_weights = {
            "W_N": 5, "W_E": 10, "W_S": 5,
            "N_W": 15, "N_E": 5, "N_S": 15,
            "E_N": 15, "E_S": 5, "E_W": 10,
            "S_N": 15, "S_E": 5, "S_W": 5
        }

        # Initialize server communication if URL is provided
        self._server_url = server_url
        self._agent_id = agent_id
        if server_url:
            self._communicator = AgentCommunicatorTesting(server_url, agent_id, mapping_config, env_file_path)
            self._communicator.update_status("test_initialized")
            self._communicator.update_config({
                "max_steps": max_steps,
                "green_duration": green_duration,
                "yellow_duration": yellow_duration,
                "num_states": num_states,
                "num_actions": num_actions,
                "mode": "interactive_testing"
            })
            self._communicator.start_background_sync()
        else:
            self._communicator = None

    @property
    def reward_episode(self):
        """Get the reward episode data"""
        return self._reward_episode

    @property
    def queue_length_episode(self):
        """Get the queue length episode data"""
        return self._queue_length_episode

    def run(self, episode):
        """
        Runs the interactive testing simulation with UI and reports to server if enabled
        """
        start_time = timeit.default_timer()

        if self._communicator:
            self._communicator.update_status("testing")

        # Reset episode arrays
        self._reward_episode = []
        self._queue_length_episode = []

        # Create and show the main window
        app = QApplication.instance()
        if not app:
            app = QApplication([])
        self.window = MainWindow()

        # Connect signals to window slots
        self.step_updated.connect(self.window.update_step)
        self.vehicle_updated.connect(self.window.update_vehicles)
        self.stats_updated.connect(self.window.update_statistics)
        self.cumulative_stats_updated.connect(self.window.update_cumulative_statistics)
        self.window.auto_spawn_check.stateChanged.connect(self.toggle_auto_spawn)

        # Connect the window's start button to our simulation control
        self.window.start_button.clicked.connect(self.toggle_simulation)

        # Connect preset selection to our distribution update method
        for i, radio in enumerate(self.window.preset_buttons):
            radio.clicked.connect(lambda checked, idx=i+1: self.update_distribution_preset(idx) if checked else None)

        # Show window and process events
        self.window.show()
        app.processEvents()

        # Initialize simulation variables
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1  # dummy init

        # Get initial sync timing if available
        if self._communicator:
            sync_data = self._communicator.get_sync_timing()
            if sync_data:
                self._adjust_timing(sync_data)

        # Main simulation loop
        while self._step < self._max_steps:
            # Process Qt events to keep UI responsive
            app.processEvents()

            # Only run simulation if it's active
            if self.running and traci.isLoaded():
                try:
                    # Handle automatic spawning with random intervals
                    if self.auto_spawn:
                        if self.spawn_interval_random:
                            current_interval = random.randint(self.min_interval, self.max_interval)
                        else:
                            current_interval = self.spawn_interval

                        if (self._step - self.last_spawn_step) >= current_interval:
                            self.last_spawn_step = self._step
                            # Calculate count before using it
                            if self.spawn_count_random:
                                count = random.randint(self.min_count, self.max_count)
                            else:
                                count = self.spawn_count
                            
                            # Spawn vehicles
                            for _ in range(count):
                                self.spawn_random_vehicle()
                            print(f"Spawned {count} vehicles")
                    else:
                        print("Auto spawn is disabled")

                    # Get current state of the intersection
                    current_state = self._get_state()

                    # Calculate reward of previous action
                    current_total_wait = self._collect_waiting_times()
                    reward = old_total_wait - current_total_wait

                    # Choose the light phase to activate
                    action = self._choose_action(current_state)

                    # If the chosen phase is different from the last phase, activate the yellow phase
                    if self._step != 0 and old_action != action:
                        self._set_yellow_phase(old_action)
                        self._simulate(self._yellow_duration)

                    # Execute the green phase
                    self._set_green_phase(action)
                    self._simulate(self._green_duration)

                    # Save variables for next step
                    old_action = action
                    old_total_wait = current_total_wait

                    # Add reward to episode total
                    self._reward_episode.append(reward)

                    # Update UI with current data
                    self.step_updated.emit(self._step)
                    self.vehicle_updated.emit(self.get_vehicle_data())
                    self.stats_updated.emit(self.get_statistics())
                    self.cumulative_stats_updated.emit(self.get_cumulative_statistics())

                    # Track vehicles and handle incoming vehicles
                    self._track_vehicles()
                    self._check_incoming_vehicles()

                    # Update server with state and get new sync timing
                    if self._communicator:
                        # Send current state
                        self._communicator.send_state(current_state, self._step, {
                            'queue_length': self._get_queue_length(),
                            'current_phase': traci.trafficlight.getPhase("TL"),
                            'incoming_vehicles': {
                                'N': traci.edge.getLastStepVehicleNumber("N2TL"),
                                'S': traci.edge.getLastStepVehicleNumber("S2TL"),
                                'E': traci.edge.getLastStepVehicleNumber("E2TL"),
                                'W': traci.edge.getLastStepVehicleNumber("W2TL")
                            },
                            'avg_speed': {
                                'N': traci.edge.getLastStepMeanSpeed("N2TL"),
                                'S': traci.edge.getLastStepMeanSpeed("S2TL"),
                                'E': traci.edge.getLastStepMeanSpeed("E2TL"),
                                'W': traci.edge.getLastStepMeanSpeed("W2TL")
                            }
                        })

                        # Get new sync timing periodically
                        if self._step % 60 == 0:  # Check for new sync timing every minute
                            sync_data = self._communicator.get_sync_timing()
                            if sync_data:
                                self._adjust_timing(sync_data)

                except traci.exceptions.FatalTraCIError as e:
                    print(f"TraCI error: {e}")
                    break
                except Exception as e:
                    print(f"Error in simulation loop: {e}")
                    break

            # Small delay to prevent CPU overuse
            time.sleep(0.01)

        # End simulation
        self.running = False
        if traci.isLoaded():
            traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        # Report final results to server
        if self._communicator:
            total_reward = np.sum(self._reward_episode)
            avg_queue_length = np.mean(self._queue_length_episode) if self._queue_length_episode else 0
            total_waiting_time = np.sum(self._queue_length_episode) if self._queue_length_episode else 0

            self._communicator.update_episode_result(
                episode=episode,
                reward=total_reward,
                queue_length=avg_queue_length,
                waiting_time=total_waiting_time
            )
            self._communicator.update_status("test_completed")

        return simulation_time

    def toggle_simulation(self):
        """Toggle simulation on/off"""
        if not self.running:
            # Start simulation
            self.running = True
            try:
                # Initialize SUMO if not already running
                if not traci.isLoaded():
                    try:
                        traci.start(self._sumo_cmd)
                        time.sleep(2)  # Wait for traci to be ready
                    except traci.exceptions.FatalTraCIError as e:
                        print(f"Error starting SUMO: {e}")
                        self.running = False
                        self.window.start_button.setText("Start Simulation")
                        self.window.status_label.setText("Status: Error starting SUMO")
                        return
                    except Exception as e:
                        print(f"Unexpected error starting SUMO: {e}")
                        self.running = False
                        self.window.start_button.setText("Start Simulation")
                        self.window.status_label.setText("Status: Error starting SUMO")
                        return

                self.window.start_button.setText("Stop Simulation")
                self.window.status_label.setText("Status: Running")
            except Exception as e:
                print(f"Error starting simulation: {e}")
                self.running = False
                self.window.start_button.setText("Start Simulation")
                self.window.status_label.setText("Status: Error starting simulation")
        else:
            # Stop simulation
            self.running = False
            self.window.start_button.setText("Start Simulation")
            self.window.status_label.setText("Status: Stopped")
            # Close SUMO
            if traci.isLoaded():
                try:
                    traci.close()
                except traci.exceptions.FatalTraCIError as e:
                    print(f"Error closing SUMO: {e}")
                except Exception as e:
                    print(f"Unexpected error closing SUMO: {e}")

    def spawn_random_vehicle(self):
        """Spawn a random vehicle in the simulation"""
        try:
            # Select vehicle type based on distribution
            vehicle_type = random.choices(
                list(self.vehicle_types.keys()),
                weights=list(self.vehicle_types.values())
            )[0]

            # Select route based on distribution
            route = random.choices(
                list(self.route_weights.keys()),
                weights=list(self.route_weights.values())
            )[0]

            # Get speed range for vehicle type
            min_speed, max_speed = self.speed_ranges[vehicle_type]
            speed = random.uniform(min_speed, max_speed)

            # Create unique vehicle ID using counter and larger random number
            self._vehicle_counter += 1
            random_suffix = random.randint(10000, 99999)  # Increased range
            vehicle_id = f"{vehicle_type}_{route}_{self._vehicle_counter}_{random_suffix}"

            # Add vehicle with proper type and departure time
            traci.vehicle.add(
                vehID=vehicle_id,
                routeID=route,
                typeID=vehicle_type,
                departLane="random",
                departSpeed=str(speed)
            )
        except Exception as e:
            print(f"Error spawning random vehicle: {e}")

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo
        """
        if not traci.isLoaded():
            return np.zeros(self._num_states)

        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)

            # x is the position of the car in the lane
            lane_cell = int(lane_pos / 7.5)

            # Find the lane where the car is located
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 0 and lane_group <= 7:
                # Calculate position in state array
                # Each lane group has 10 cells (0-9)
                car_position = (lane_group * 10) + min(lane_cell, 9)
                if car_position < self._num_states:
                    state[car_position] = 1

        return state

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        if not traci.isLoaded():
            return 0

        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state):
        """
        Pick the best action known for the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        if not traci.isLoaded():
            return

        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)
        traci.trafficlight.setPhaseDuration("TL", self.phase_durations[yellow_phase_code])

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if not traci.isLoaded():
            return

        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
            traci.trafficlight.setPhaseDuration("TL", self.phase_durations[PHASE_NS_GREEN])
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
            traci.trafficlight.setPhaseDuration("TL", self.phase_durations[PHASE_NSL_GREEN])
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
            traci.trafficlight.setPhaseDuration("TL", self.phase_durations[PHASE_EW_GREEN])
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
            traci.trafficlight.setPhaseDuration("TL", self.phase_durations[PHASE_EWL_GREEN])

    def _track_vehicles(self):
        """Track vehicles that enter and exit the simulation"""
        if not traci.isLoaded():
            return

        try:
            # Get current vehicles
            current_vehicles = set(traci.vehicle.getIDList())
            
            # Find vehicles that have exited
            exited = self._active_vehicles - current_vehicles
            for vehicle_id in exited:
                try:
                    # Get the last road the vehicle was on
                    last_road = traci.vehicle.getRoadID(vehicle_id)
                    last_position = traci.vehicle.getPosition(vehicle_id)
                    last_speed = traci.vehicle.getSpeed(vehicle_id)
                    last_lane = traci.vehicle.getLaneIndex(vehicle_id)
                    
                    # Determine if vehicle exited through a boundary
                    is_boundary_exit = False
                    exit_direction = None
                    
                    # Check if vehicle exited through a boundary road
                    if last_road in self.roads.values():
                        # Get the road's position and dimensions
                        road_shape = traci.edge.getShape(last_road)
                        if road_shape:
                            start_pos, end_pos = road_shape[0], road_shape[-1]
                            
                            # Calculate distance to road endpoints
                            dist_to_start = ((last_position[0] - start_pos[0])**2 + 
                                           (last_position[1] - start_pos[1])**2)**0.5
                            dist_to_end = ((last_position[0] - end_pos[0])**2 + 
                                         (last_position[1] - end_pos[1])**2)**0.5
                            
                            # If vehicle is close to either endpoint, consider it a boundary exit
                            if dist_to_start < 5.0 or dist_to_end < 5.0:
                                is_boundary_exit = True
                                # Determine exit direction based on road and position
                                if last_road == 'N2TL':
                                    exit_direction = 'north'
                                elif last_road == 'S2TL':
                                    exit_direction = 'south'
                                elif last_road == 'E2TL':
                                    exit_direction = 'east'
                                elif last_road == 'W2TL':
                                    exit_direction = 'west'
                    
                    # Store vehicle info
                    self._exited_vehicles[vehicle_id] = {
                        'type': traci.vehicle.getTypeID(vehicle_id),
                        'route': traci.vehicle.getRouteID(vehicle_id),
                        'speed': last_speed,
                        'lane': last_lane,
                        'position': last_position,
                        'is_boundary_exit': is_boundary_exit,
                        'exit_direction': exit_direction,
                        'destination': self.road_connections.get(last_road)
                    }
                    
                    # Log vehicle exit
                    print(f"Vehicle {vehicle_id} exited through {'boundary' if is_boundary_exit else 'internal'} " +
                          f"at {last_position} on road {last_road}")
                    
                    # Send vehicle info to server if connected and it's a boundary exit
                    if self._communicator and is_boundary_exit and exit_direction:
                        self._communicator.send_state(None, self._step, {
                            'vehicle_transfer': {
                                'vehicle_id': vehicle_id,
                                'type': traci.vehicle.getTypeID(vehicle_id),
                                'route': traci.vehicle.getRouteID(vehicle_id),
                                'speed': last_speed,
                                'lane': last_lane,
                                'position': last_position,
                                'exit_direction': exit_direction,
                                'from_agent': self._agent_id,
                                'to_agent': self.road_connections.get(last_road)
                            }
                        })
                except traci.exceptions.TraCIException:
                    # Vehicle is no longer in simulation, skip it
                    continue
                except Exception as e:
                    print(f"Error tracking exited vehicle {vehicle_id}: {e}")
            
            # Update active vehicles
            self._active_vehicles = current_vehicles
        except Exception as e:
            print(f"Error in _track_vehicles: {e}")

    def _check_incoming_vehicles(self):
        """Check for and spawn incoming vehicles from other intersections"""
        if not traci.isLoaded() or not self._communicator:
            return

        try:
            # Get any incoming vehicles from the server
            response = self._communicator.get_coordination_data()
            if response and 'incoming_vehicles' in response:
                for vehicle_data in response['incoming_vehicles']:
                    if vehicle_data['to_agent'] == self._agent_id:
                        # Add to incoming vehicles list with spawn position
                        entry_road = None
                        entry_lane = 0
                        
                        # Determine entry road based on exit direction from previous intersection
                        if vehicle_data.get('exit_direction'):
                            if vehicle_data['exit_direction'] == 'north':
                                entry_road = 'S2TL'
                            elif vehicle_data['exit_direction'] == 'south':
                                entry_road = 'N2TL'
                            elif vehicle_data['exit_direction'] == 'east':
                                entry_road = 'W2TL'
                            elif vehicle_data['exit_direction'] == 'west':
                                entry_road = 'E2TL'
                        
                        if entry_road:
                            # Get road shape to determine spawn position
                            road_shape = traci.edge.getShape(entry_road)
                            if road_shape:
                                # Spawn at the start of the road
                                spawn_pos = road_shape[0]
                                
                                # Add spawn information to vehicle data
                                vehicle_data['spawn_road'] = entry_road
                                vehicle_data['spawn_lane'] = entry_lane
                                vehicle_data['spawn_position'] = spawn_pos
                                
                                # Add to incoming vehicles list
                                self._incoming_vehicles.append(vehicle_data)
                                print(f"Queued incoming vehicle {vehicle_data['vehicle_id']} for spawn on {entry_road}")
            
            # Spawn any incoming vehicles
            while self._incoming_vehicles:
                vehicle_data = self._incoming_vehicles.pop(0)
                try:
                    # Create route for the vehicle
                    route_id = f"route_{vehicle_data['vehicle_id']}"
                    route_edges = [vehicle_data['spawn_road']]
                    
                    # Add destination edge based on original route
                    if 'route' in vehicle_data:
                        route_parts = vehicle_data['route'].split()
                        if len(route_parts) > 1:
                            route_edges.append(route_parts[1])
                    
                    # Add the route
                    traci.route.add(route_id, route_edges)
                    
                    # Spawn the vehicle
                    traci.vehicle.add(
                        vehID=vehicle_data['vehicle_id'],
                        routeID=route_id,
                        typeID=vehicle_data['type'],
                        departLane=str(vehicle_data['spawn_lane']),
                        departSpeed=str(vehicle_data['speed']),
                        departPos="0"
                    )
                    
                    print(f"Spawned incoming vehicle {vehicle_data['vehicle_id']} on {vehicle_data['spawn_road']}")
                    
                except Exception as e:
                    print(f"Error spawning incoming vehicle: {e}")
                    # Put the vehicle back in the queue if there was an error
                    self._incoming_vehicles.insert(0, vehicle_data)
                    break
        except Exception as e:
            print(f"Error checking incoming vehicles: {e}")

    def _simulate(self, steps_todo):
        """
        Proceed in the simulation in sumo
        """
        if not traci.isLoaded():
            return

        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            try:
                traci.simulationStep()  # simulate 1 step in sumo
                self._step += 1  # update the step counter
                steps_todo -= 1
                queue_length = self._get_queue_length()
                self._queue_length_episode.append(queue_length)
                
                # Only handle vehicle tracking and spawning if auto_spawn is enabled
                if self.auto_spawn:
                    # Track vehicles and handle incoming vehicles
                    self._track_vehicles()
                    self._check_incoming_vehicles()
                
                # Emit step update signal
                self.step_updated.emit(self._step)
            except traci.exceptions.FatalTraCIError as e:
                print(f"TraCI error during simulation step: {e}")
                self.running = False
                self.window.start_button.setText("Start Simulation")
                self.window.status_label.setText("Status: Error - Connection lost")
                break
            except Exception as e:
                print(f"Unexpected error during simulation step: {e}")
                self.running = False
                self.window.start_button.setText("Start Simulation")
                self.window.status_label.setText("Status: Error - Unexpected error")
                break

    def _get_queue_length(self):
        """
        Calculate the total number of cars with speed = 0 in every incoming lane
        """
        if not traci.isLoaded():
            return 0

        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _adjust_timing(self, sync_data):
        """
        Adjust simulation timing based on server sync data
        """
        if 'timing' in sync_data:
            timing = sync_data['timing']
            if 'green_duration' in timing:
                self._green_duration = timing['green_duration']
            if 'yellow_duration' in timing:
                self._yellow_duration = timing['yellow_duration']

    def cleanup(self):
        """Clean up when done"""
        # Stop the simulation
        self.running = False
        
        # Close SUMO if it's running
        if traci.isLoaded():
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError as e:
                print(f"Error closing SUMO: {e}")
            except Exception as e:
                print(f"Unexpected error closing SUMO: {e}")
        
        # Update server status if connected
        if self._communicator:
            try:
                self._communicator.update_status("test_terminated")
                self._communicator.stop_background_sync()
                self._communicator.sync_with_server()  # Final sync
            except Exception as e:
                print(f"Error updating server status: {e}")

    def closeEvent(self, event):
        """Handle window close event"""
        self.cleanup()
        event.accept()

    def update_distribution_preset(self, preset_num):
        """Update vehicle type and route distributions based on preset number"""
        if preset_num == 1:  # Urban Rush Hour
            self.vehicle_types = {
                "veh_passenger": 75,
                "veh_bus": 15,
                "veh_truck": 5,
                "veh_emergency": 3,
                "veh_motorcycle": 2
            }
            self.route_weights = {
                "W_N": 12, "W_E": 15, "W_S": 8,
                "N_W": 8, "N_E": 15, "N_S": 12,
                "E_N": 15, "E_S": 8, "E_W": 12,
                "S_N": 8, "S_E": 12, "S_W": 15
            }
        elif preset_num == 2:  # Highway Traffic
            self.vehicle_types = {
                "veh_passenger": 45,
                "veh_bus": 10,
                "veh_truck": 35,
                "veh_emergency": 5,
                "veh_motorcycle": 5
            }
            self.route_weights = {
                "W_N": 10, "W_E": 20, "W_S": 10,
                "N_W": 10, "N_E": 20, "N_S": 10,
                "E_N": 10, "E_S": 20, "E_W": 10,
                "S_N": 10, "S_E": 20, "S_W": 10
            }
        elif preset_num == 3:  # Mixed Traffic
            self.vehicle_types = {
                "veh_passenger": 40,
                "veh_bus": 20,
                "veh_truck": 20,
                "veh_emergency": 10,
                "veh_motorcycle": 10
            }
            self.route_weights = {
                "W_N": 8, "W_E": 8, "W_S": 8,
                "N_W": 8, "N_E": 8, "N_S": 8,
                "E_N": 8, "E_S": 8, "E_W": 8,
                "S_N": 8, "S_E": 8, "S_W": 8
            }
        elif preset_num == 4:  # Emergency Heavy
            self.vehicle_types = {
                "veh_passenger": 30,
                "veh_bus": 10,
                "veh_truck": 10,
                "veh_emergency": 40,
                "veh_motorcycle": 10
            }
            self.route_weights = {
                "W_N": 15, "W_E": 5, "W_S": 15,
                "N_W": 5, "N_E": 15, "N_S": 5,
                "E_N": 15, "E_S": 5, "E_W": 15,
                "S_N": 5, "S_E": 15, "S_W": 5
            }
        elif preset_num == 5:  # North-South Dominant
            self.vehicle_types = {
                "veh_passenger": 45,
                "veh_bus": 10,
                "veh_truck": 2,
                "veh_emergency": 3,
                "veh_motorcycle": 40
            }
            self.route_weights = {
                "W_N": 5, "W_E": 10, "W_S": 5,
                "N_W": 15, "N_E": 5, "N_S": 15,
                "E_N": 15, "E_S": 5, "E_W": 10,
                "S_N": 15, "S_E": 5, "S_W": 5
            }
        elif preset_num == 6:  # East-West Dominant
            self.vehicle_types = {
                "veh_passenger": 55,
                "veh_bus": 12,
                "veh_truck": 1,
                "veh_emergency": 2,
                "veh_motorcycle": 30
            }
            self.route_weights = {
                "W_N": 10, "W_E": 15, "W_S": 10,
                "N_W": 5, "N_E": 15, "N_S": 5,
                "E_N": 5, "E_S": 15, "E_W": 15,
                "S_N": 5, "S_E": 15, "S_W": 5
            }
        elif preset_num == 7:  # Diagonal Dominant
            self.vehicle_types = {
                "veh_passenger": 40,
                "veh_bus": 8,
                "veh_truck": 2,
                "veh_emergency": 5,
                "veh_motorcycle": 45
            }
            self.route_weights = {
                "W_N": 15, "W_E": 5, "W_S": 5,
                "N_W": 5, "N_E": 15, "N_S": 5,
                "E_N": 5, "E_S": 15, "E_W": 5,
                "S_N": 5, "S_E": 5, "S_W": 15
            }
        elif preset_num == 8:  # Circular Flow
            self.vehicle_types = {
                "veh_passenger": 35,
                "veh_bus": 7,
                "veh_truck": 1,
                "veh_emergency": 2,
                "veh_motorcycle": 55
            }
            self.route_weights = {
                "W_N": 15, "W_E": 5, "W_S": 5,
                "N_W": 5, "N_E": 15, "N_S": 5,
                "E_N": 5, "E_S": 15, "E_W": 5,
                "S_N": 5, "S_E": 5, "S_W": 15
            }

    def get_vehicle_data(self):
        """Get current vehicle data for UI update"""
        if not traci.isLoaded():
            return {}

        vehicles = {}
        for vid in traci.vehicle.getIDList():
            try:
                vehicles[vid] = {
                    'route': traci.vehicle.getRouteID(vid),
                    'road': traci.vehicle.getRoadID(vid),
                    'lane': traci.vehicle.getLaneIndex(vid),
                    'speed': round(traci.vehicle.getSpeed(vid), 1),
                    'waiting': round(traci.vehicle.getWaitingTime(vid), 1)
                }
            except Exception as e:
                print(f"Error getting vehicle data for {vid}: {e}")

        return vehicles

    def get_statistics(self):
        """Get current statistics for UI update"""
        if not traci.isLoaded():
            return {}

        stats = {}
        directions = {
            "north": "N2TL",
            "south": "S2TL",
            "east": "E2TL",
            "west": "W2TL"
        }

        for direction, edge in directions.items():
            stats[direction] = {
                'count': traci.edge.getLastStepVehicleNumber(edge),
                'queue': traci.edge.getLastStepHaltingNumber(edge),
                'speed': max(0, traci.edge.getLastStepMeanSpeed(edge))
            }

        try:
            stats['light_phase'] = traci.trafficlight.getPhase("TL")
        except:
            stats['light_phase'] = -1

        return stats

    def moving_average(self, data, window_size=5):
        """Calculate moving average of data with given window size"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def get_cumulative_statistics(self):
        """Get cumulative statistics for UI update"""
        try:
            if not traci.isLoaded():
                return {
                    'total_queue': 0,
                    'total_waiting_time': 0,
                    'total_vehicles': 0,
                    'max_queue': 0,
                    'max_waiting_time': 0,
                    'total_length': 0,
                    'average_queue': 0,
                    'average_waiting_time': 0,
                    'average_length': 0,
                    'road_stats': {
                        'N2TL': self._get_empty_road_stats(),
                        'S2TL': self._get_empty_road_stats(),
                        'E2TL': self._get_empty_road_stats(),
                        'W2TL': self._get_empty_road_stats()
                    },
                    'current_step': self._step
                }

            # Get current vehicle list
            vehicle_list = traci.vehicle.getIDList()

            # Calculate total statistics
            total_queue = sum(self._queue_length_episode) if self._queue_length_episode else 0
            total_waiting_time = sum(self._waiting_times.values()) if self._waiting_times else 0
            total_vehicles = len(vehicle_list)
            max_queue = max(self._queue_length_episode) if self._queue_length_episode else 0
            max_waiting_time = max(self._waiting_times.values()) if self._waiting_times else 0

            # Calculate averages
            avg_queue = np.mean(self._queue_length_episode) if self._queue_length_episode else 0
            avg_waiting_time = np.mean(list(self._waiting_times.values())) if self._waiting_times else 0

            # Initialize arrays for plot data
            steps = np.array(range(self._step + 1))
            queue_data = np.zeros(self._step + 1)
            wait_data = np.zeros(self._step + 1)
            length_data = np.zeros(self._step + 1)

            # Fill arrays with actual data
            for i, q in enumerate(self._queue_length_episode):
                if i < len(queue_data):
                    queue_data[i] = q

            stats = {
                'total_queue': total_queue,
                'total_waiting_time': total_waiting_time,
                'total_vehicles': total_vehicles,
                'max_queue': max_queue,
                'max_waiting_time': max_waiting_time,
                'total_length': 0,
                'average_queue': avg_queue,
                'average_waiting_time': avg_waiting_time,
                'average_length': 0,
                'road_stats': {},
                'current_step': self._step,
                'plot_data': {
                    'steps': steps,
                    'queue': queue_data,
                    'wait': wait_data,
                    'length': length_data
                }
            }

            # Add per-road statistics
            for road_id in ["N2TL", "S2TL", "E2TL", "W2TL"]:
                try:
                    # Get vehicles on this road
                    road_vehicles = [vid for vid in vehicle_list if traci.vehicle.getRoadID(vid) == road_id]

                    # Calculate road-specific statistics
                    current_queue = traci.edge.getLastStepHaltingNumber(road_id)
                    current_vehicles = traci.edge.getLastStepVehicleNumber(road_id)
                    current_waiting_time = sum(traci.vehicle.getWaitingTime(vid) for vid in road_vehicles)

                    # Calculate averages
                    avg_wait = np.mean([traci.vehicle.getWaitingTime(vid) for vid in road_vehicles]) if road_vehicles else 0

                    stats['road_stats'][road_id] = {
                        'total_queue': current_queue,
                        'total_waiting_time': current_waiting_time,
                        'total_vehicles': current_vehicles,
                        'max_queue': current_queue,
                        'max_waiting_time': max((traci.vehicle.getWaitingTime(vid) for vid in road_vehicles), default=0),
                        'total_length': 0,
                        'current_queue': current_queue,
                        'current_waiting_time': current_waiting_time,
                        'current_vehicles': current_vehicles,
                        'current_length': 0,
                        'average_queue': current_queue,
                        'average_waiting_time': avg_wait,
                        'average_length': 0
                    }
                except Exception as e:
                    print(f"Error calculating statistics for road {road_id}: {e}")
                    stats['road_stats'][road_id] = self._get_empty_road_stats()

            return stats

        except Exception as e:
            print(f"Error in get_cumulative_statistics: {e}")
            return {
                'total_queue': 0,
                'total_waiting_time': 0,
                'total_vehicles': 0,
                'max_queue': 0,
                'max_waiting_time': 0,
                'total_length': 0,
                'average_queue': 0,
                'average_waiting_time': 0,
                'average_length': 0,
                'road_stats': {
                    'N2TL': self._get_empty_road_stats(),
                    'S2TL': self._get_empty_road_stats(),
                    'E2TL': self._get_empty_road_stats(),
                    'W2TL': self._get_empty_road_stats()
                },
                'current_step': self._step,
                'plot_data': {
                    'steps': np.array([0]),
                    'queue': np.array([0]),
                    'wait': np.array([0]),
                    'length': np.array([0])
                }
            }

    def _get_empty_road_stats(self):
        """Helper method to create empty road statistics"""
        return {
            'total_queue': 0,
            'total_waiting_time': 0,
            'total_vehicles': 0,
            'max_queue': 0,
            'max_waiting_time': 0,
            'total_length': 0,
            'current_queue': 0,
            'current_waiting_time': 0,
            'current_vehicles': 0,
            'current_length': 0,
            'average_queue': 0,
            'average_waiting_time': 0,
            'average_length': 0
        } 
    
    def toggle_auto_spawn(self, state):
        """Toggle auto-spawn state based on checkbox"""
        self.auto_spawn = bool(state)
        print(f"Auto-spawn {'enabled' if self.auto_spawn else 'disabled'}") 