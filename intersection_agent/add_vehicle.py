import os
import sys
import traci
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QLineEdit, QTableWidget, QTableWidgetItem, QGroupBox,
                            QCheckBox, QSlider, QSpinBox, QRadioButton, QFrame, QHeaderView,
                            QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Import sumo utilities
from sumolib import checkBinary

class SimulationThread(QThread):
    step_updated = pyqtSignal(int)
    vehicle_updated = pyqtSignal(dict)
    stats_updated = pyqtSignal(dict)
    cumulative_stats_updated = pyqtSignal(dict)
    
    def __init__(self, sumo_cmd=None):
        super().__init__()
        self.running = False
        self.step = 0
        self.speed = 5.0
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
        self._sumo_cmd = sumo_cmd
        
        # Road IDs
        self.roads = {
            'north': 'N2TL',
            'south': 'S2TL',
            'east': 'E2TL',
            'west': 'W2TL'
        }
        
        # Cumulative statistics
        self.total_queue = 0
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.max_queue = 0
        self.max_waiting_time = 0
        self.vehicle_lengths = {
            "veh_passenger": 5.0,
            "veh_bus": 12.0,
            "veh_truck": 10.0,
            "veh_emergency": 6.0,
            "veh_motorcycle": 2.0
        }
        self.total_length = 0
        
        # Per-road statistics
        self.road_stats = {}
        for road in self.roads.values():
            self.road_stats[road] = {
                'total_queue': 0,
                'total_waiting_time': 0,
                'total_vehicles': 0,
                'max_queue': 0,
                'max_waiting_time': 0,
                'total_length': 0,
                'current_queue': 0,
                'current_waiting_time': 0,
                'current_vehicles': 0,
                'current_length': 0
            }
        
        # Vehicle type distribution
        self.vehicle_types = {
            "veh_passenger": 70,    # 70% standard cars
            "veh_bus": 10,          # 10% buses
            "veh_truck": 10,        # 10% trucks
            "veh_emergency": 5,     # 5% emergency vehicles
            "veh_motorcycle": 5     # 5% motorcycles
        }
        
        # Route distribution
        self.route_weights = {
            "W_N": 15, "W_E": 15, "W_S": 15,
            "N_W": 15, "N_E": 15, "N_S": 15,
            "E_N": 15, "E_S": 15, "E_W": 15,
            "S_N": 15, "S_E": 15, "S_W": 15
        }
        
        # Speed ranges for different vehicle types (reduced to prevent too high speeds)
        self.speed_ranges = {
            "veh_passenger": (3, 8),    # Reduced from (5, 15)
            "veh_bus": (2, 6),          # Reduced from (3, 10)
            "veh_truck": (2, 5),        # Reduced from (3, 8)
            "veh_emergency": (4, 10),   # Reduced from (8, 20)
            "veh_motorcycle": (3, 8)    # Reduced from (7, 18)
        }
    
    def run(self):
        try:
            # Only start SUMO if we have a command and traci isn't already running
            if self._sumo_cmd and not traci.isLoaded():
                traci.start(self._sumo_cmd)
                time.sleep(1)  # Wait for traci to be ready
            
            # Define phase durations (in seconds)
            phase_durations = {
                0: 31,  # NS Green
                1: 2,   # NS Yellow
                2: 15,  # NSL Green
                3: 2,   # NSL Yellow
                4: 31,  # EW Green
                5: 2,   # EW Yellow
                6: 15,  # EWL Green
                7: 2    # EWL Yellow
            }
            
            current_phase = -1
            
            while self.running:
                step_duration = 1.0 / self.speed
                traci.simulationStep()
                self.step += 1
                
                # Check if phase changed
                try:
                    new_phase = traci.trafficlight.getPhase("TL")
                    if new_phase != current_phase:
                        current_phase = new_phase
                        # Set new phase duration
                        traci.trafficlight.setPhaseDuration("TL", phase_durations[new_phase])
                except Exception as e:
                    print(f"Error setting traffic light phase: {e}")
                
                # Handle automatic spawning with random intervals
                if self.auto_spawn:
                    if self.spawn_interval_random:
                        current_interval = random.randint(self.min_interval, self.max_interval)
                    else:
                        current_interval = self.spawn_interval
                        
                    if (self.step - self.last_spawn_step) >= current_interval:
                        self.last_spawn_step = self.step
                        if self.spawn_count_random:
                            count = random.randint(self.min_count, self.max_count)
                        else:
                            count = self.spawn_count
                        for _ in range(count):
                            self.spawn_random_vehicle()
                
                # Update cumulative statistics
                self.update_cumulative_statistics()
                
                # Emit signals
                self.step_updated.emit(self.step)
                self.vehicle_updated.emit(self.get_vehicle_data())
                self.stats_updated.emit(self.get_statistics())
                self.cumulative_stats_updated.emit(self.get_cumulative_statistics())
                
                time.sleep(step_duration)
                
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            # Don't close traci here - let the main simulation handle that
            pass
    
    def get_vehicle_data(self):
        if 'traci' not in sys.modules or not traci.isLoaded():
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
        if 'traci' not in sys.modules or not traci.isLoaded():
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
    
    def get_cumulative_statistics(self):
        stats = {
            'total_queue': self.total_queue,
            'total_waiting_time': self.total_waiting_time,
            'total_vehicles': self.total_vehicles,
            'max_queue': self.max_queue,
            'max_waiting_time': self.max_waiting_time,
            'total_length': self.total_length,
            'average_queue': self.total_queue / max(1, self.step),
            'average_waiting_time': self.total_waiting_time / max(1, self.total_vehicles),
            'average_length': self.total_length / max(1, self.total_vehicles),
            'road_stats': {}
        }
        
        # Add per-road statistics
        for road_id, road_data in self.road_stats.items():
            stats['road_stats'][road_id] = {
                'total_queue': road_data['total_queue'],
                'total_waiting_time': road_data['total_waiting_time'],
                'total_vehicles': road_data['total_vehicles'],
                'max_queue': road_data['max_queue'],
                'max_waiting_time': road_data['max_waiting_time'],
                'total_length': road_data['total_length'],
                'current_queue': road_data['current_queue'],
                'current_waiting_time': road_data['current_waiting_time'],
                'current_vehicles': road_data['current_vehicles'],
                'current_length': road_data['current_length'],
                'average_queue': road_data['total_queue'] / max(1, self.step),
                'average_waiting_time': road_data['total_waiting_time'] / max(1, road_data['total_vehicles']),
                'average_length': road_data['total_length'] / max(1, road_data['total_vehicles'])
            }
        
        return stats
    
    def spawn_random_vehicle(self):
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
            
            # Random lane selection
            lane = random.choice(["random", "0", "1", "2"])
            
            # Create unique vehicle ID using timestamp and random number
            timestamp = int(time.time() * 1000)  # milliseconds
            random_suffix = random.randint(1000, 9999)
            vehicle_id = f"auto_{vehicle_type}_{route}_{timestamp}_{random_suffix}"
            
            traci.vehicle.add(
                vehID=vehicle_id,
                routeID=route,
                typeID=vehicle_type,
                departLane=lane,
                departSpeed=str(speed)
            )
        except Exception as e:
            print(f"Error spawning random vehicle: {e}")
    
    def update_cumulative_statistics(self):
        if 'traci' not in sys.modules or not traci.isLoaded():
            return
        
        try:
            # Reset current road statistics
            for road in self.roads.values():
                self.road_stats[road]['current_queue'] = 0
                self.road_stats[road]['current_waiting_time'] = 0
                self.road_stats[road]['current_vehicles'] = 0
                self.road_stats[road]['current_length'] = 0
            
            # Get current vehicles
            vehicles = traci.vehicle.getIDList()
            current_queue = 0
            current_waiting_time = 0
            current_length = 0
            
            for vid in vehicles:
                try:
                    # Get vehicle info
                    road_id = traci.vehicle.getRoadID(vid)
                    waiting_time = traci.vehicle.getWaitingTime(vid)
                    vehicle_type = traci.vehicle.getTypeID(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    
                    # Update road-specific statistics
                    if road_id in self.road_stats:
                        self.road_stats[road_id]['current_vehicles'] += 1
                        self.road_stats[road_id]['current_waiting_time'] += waiting_time
                        if speed < 0.1:
                            self.road_stats[road_id]['current_queue'] += 1
                        if vehicle_type in self.vehicle_lengths:
                            self.road_stats[road_id]['current_length'] += self.vehicle_lengths[vehicle_type]
                    
                    # Update global statistics
                    if speed < 0.1:
                        current_queue += 1
                    current_waiting_time += waiting_time
                    if vehicle_type in self.vehicle_lengths:
                        current_length += self.vehicle_lengths[vehicle_type]
                
                except Exception as e:
                    print(f"Error processing vehicle {vid}: {e}")
            
            # Update cumulative road statistics
            for road in self.roads.values():
                stats = self.road_stats[road]
                stats['total_queue'] += stats['current_queue']
                stats['total_waiting_time'] += stats['current_waiting_time']
                stats['total_vehicles'] += stats['current_vehicles']
                stats['total_length'] += stats['current_length']
                stats['max_queue'] = max(stats['max_queue'], stats['current_queue'])
                stats['max_waiting_time'] = max(stats['max_waiting_time'], stats['current_waiting_time'])
            
            # Update global statistics
            self.total_queue += current_queue
            self.total_waiting_time += current_waiting_time
            self.total_length += current_length
            self.total_vehicles += len(vehicles)
            self.max_queue = max(self.max_queue, current_queue)
            self.max_waiting_time = max(self.max_waiting_time, current_waiting_time)
            
        except Exception as e:
            print(f"Error updating cumulative statistics: {e}")
