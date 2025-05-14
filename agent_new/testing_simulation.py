import traci
import numpy as np
import random
import timeit
import os
import time
from agent_communicator import AgentCommunicatorTesting

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions, server_url=None, agent_id=None, mapping_config=None, env_file_path=None):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []
        self.server_url = server_url
        if server_url:
            self.communicator = AgentCommunicatorTesting(server_url, agent_id, mapping_config, env_file_path)
            self.communicator.update_status("initialized")
            self.communicator.update_config({
                "max_steps": max_steps,
                "green_duration": green_duration,
                "yellow_duration": yellow_duration,
                "num_states": num_states,
                "num_actions": num_actions,
                "mode": "testing"
            })
            self.communicator.start_background_sync()
        else:
            self.communicator = None

    def run(self, episode):
        start_time = timeit.default_timer()
        if self.communicator:
            sync_data = self.communicator.get_sync_timing()
            if sync_data:
                self._adjust_timing(sync_data)
            self.communicator.update_status("testing")
        self._reward_episode = []
        self._queue_length_episode = []
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1
        while self._step < self._max_steps:
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait
            action = self._choose_action(current_state)
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)
            self._set_green_phase(action)
            self._simulate(self._green_duration)
            old_action = action
            old_total_wait = current_total_wait
            self._reward_episode.append(reward)
            if self.communicator:
                self.communicator.send_state(current_state, self._step, {
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
                # Force immediate sync after each state update
                self.communicator.sync_with_server()
                if self._step % 60 == 0:
                    sync_data = self.communicator.get_sync_timing()
                    if sync_data:
                        self._adjust_timing(sync_data)
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        if self.communicator:
            total_reward = np.sum(self._reward_episode)
            avg_queue_length = np.mean(self._queue_length_episode) if self._queue_length_episode else 0
            total_waiting_time = np.sum(self._queue_length_episode) if self._queue_length_episode else 0
            self.communicator.update_episode_result(
                episode=episode,
                reward=total_reward,
                queue_length=avg_queue_length,
                waiting_time=total_waiting_time
            )
            self.communicator.update_status("test_completed")
            # Force final sync at the end
            self.communicator.sync_with_server()
        return simulation_time

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state):
        return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9
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
            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False
            if valid_car:
                state[car_position] = 1
        if self.communicator:
            self.communicator.send_state(state.tolist(), self._step)
        return state

    def _adjust_timing(self, sync_data):
        if not sync_data:
            return
        for target_id, timing in sync_data.items():
            if 'optimal_offset_sec' in timing:
                offset = timing['optimal_offset_sec']
                cycle_time = timing.get('cycle_time_sec', self._green_duration * 2)
                if offset > 0:
                    self._green_duration = min(self._green_duration + offset, cycle_time - self._yellow_duration)
                else:
                    self._green_duration = max(self._green_duration + offset, self._yellow_duration + 5)
                print(f"Adjusted timing for sync with {target_id}: offset={offset}s, cycle={cycle_time}s, green={self._green_duration}s")
                break

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode

    def cleanup(self):
        if self.communicator:
            self.communicator.update_status("test_terminated")
            self.communicator.stop_background_sync()
            self.communicator.sync_with_server()  # Final sync



