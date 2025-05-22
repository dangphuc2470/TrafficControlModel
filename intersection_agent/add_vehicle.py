import os
import sys
import traci
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QLineEdit, QTableWidget, QTableWidgetItem, QGroupBox,
                            QCheckBox, QSlider, QSpinBox, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import random
import time

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
    
    def __init__(self):
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
            sumo_cmd = self.set_sumo(gui=True)
            traci.start(sumo_cmd)
            
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
                
                # Emit signals
                self.step_updated.emit(self.step)
                self.vehicle_updated.emit(self.get_vehicle_data())
                self.stats_updated.emit(self.get_statistics())
                
                time.sleep(step_duration)
                
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            if 'traci' in sys.modules and traci.isLoaded():
                traci.close()
    
    def set_sumo(self, gui=True):
        if gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        
        route_file = self.set_route_file()
        
        sumo_cmd = [
            sumoBinary, 
            "-n", os.path.join('intersection', 'environment.net.xml'),
            "-r", route_file,
            "--start", "--quit-on-end=False",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--gui-settings-file", os.path.join('intersection', 'view.xml')
        ]
        
        return sumo_cmd
    
    def set_route_file(self):
        route_file = os.path.join("intersection", "interactive_routes.rou.xml")
        with open(route_file, "w") as routes:
            print("""<routes>
            <vType id="veh_passenger" vClass="passenger" color="0,191,255"/>  <!-- Bright Sky Blue for passenger cars -->
            <vType id="veh_bus" vClass="bus" color="255,69,0"/>  <!-- Bright Red-Orange for buses -->
            <vType id="veh_truck" vClass="truck" color="50,205,50"/>  <!-- Bright Lime Green for trucks -->
            <vType id="veh_emergency" vClass="emergency" color="255,215,0"/>  <!-- Bright Gold for emergency vehicles -->
            <vType id="veh_motorcycle" vClass="motorcycle" color="255,105,180"/>  <!-- Bright Pink for motorcycles -->
            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>
            <route id="S_W" edges="S2TL TL2W"/>
            </routes>""", file=routes)
        return route_file
    
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Simulation Control")
        self.setGeometry(100, 100, 1400, 800)

        # Create simulation thread FIRST
        self.sim_thread = SimulationThread()
        self.sim_thread.step_updated.connect(self.update_step)
        self.sim_thread.vehicle_updated.connect(self.update_vehicles)
        self.sim_thread.stats_updated.connect(self.update_statistics)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create left panel for auto spawn controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.create_auto_spawn_panel(left_layout)
        main_layout.addWidget(left_panel, stretch=1)

        # Create right panel for other controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.create_control_panel(right_layout)
        self.create_vehicle_panel(right_layout)
        self.create_vehicle_table(right_layout)
        self.create_statistics_panel(right_layout)
        main_layout.addWidget(right_panel, stretch=2)

        # Initialize vehicle counter
        self.vehicle_counter = 0

        # Show window
        self.show()
    
    def create_control_panel(self, parent_layout):
        group = QGroupBox("Simulation Control")
        layout = QHBoxLayout()
        
        # Start/Stop button
        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.toggle_simulation)
        layout.addWidget(self.start_button)
        
        # Render mode toggle (changed to checkbox)
        self.render_mode_check = QCheckBox("Simple Shapes")
        self.render_mode_check.setChecked(True)  # Default to simple shapes
        self.render_mode_check.stateChanged.connect(self.toggle_render_mode)
        layout.addWidget(self.render_mode_check)
        
        # Status label
        self.status_label = QLabel("Status: Not running")
        layout.addWidget(self.status_label)
        
        # Step counter
        self.step_label = QLabel("Steps: 0")
        layout.addWidget(self.step_label)
        
        # Speed control
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Simulation Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(50)  # Increased from 10 to 50
        self.speed_slider.setValue(5)  # Default to 5x speed
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        layout.addLayout(speed_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_auto_spawn_panel(self, parent_layout):
        group = QGroupBox("Auto Spawn Controls")
        layout = QVBoxLayout()
        layout.setSpacing(5)  # Reduce spacing between elements
        
        # Basic controls
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(5)
        
        # Enable/disable auto spawn
        self.auto_spawn_check = QCheckBox("Enable Auto Spawn")
        self.auto_spawn_check.setChecked(True)  # Default enabled
        self.auto_spawn_check.stateChanged.connect(self.toggle_auto_spawn)
        basic_layout.addWidget(self.auto_spawn_check)
        
        # Initialize auto spawn as enabled in the simulation thread
        self.sim_thread.auto_spawn = True  # Default active
        
        # Distribution presets
        preset_group = QGroupBox("Traffic Pattern")
        preset_layout = QVBoxLayout()
        
        # Create radio buttons for presets
        self.preset_buttons = []
        preset_names = [
            "Urban Rush Hour",
            "Highway Traffic",
            "Mixed Traffic",
            "Emergency Heavy",
            "N-S Dominant",
            "E-W Dominant",
            "Diagonal",
            "Circular"
        ]
        
        # Create radio buttons
        for i, name in enumerate(preset_names):
            radio = QRadioButton(name)
            radio.clicked.connect(lambda checked, idx=i+1: self.apply_distribution_preset(idx) if checked else None)
            self.preset_buttons.append(radio)
            preset_layout.addWidget(radio)
        
        preset_group.setLayout(preset_layout)
        basic_layout.addWidget(preset_group)
        
        # Spawn interval
        interval_group = QGroupBox("Spawn Interval")
        interval_layout = QVBoxLayout()
        interval_layout.setSpacing(2)
        
        fixed_interval_layout = QHBoxLayout()
        fixed_interval_layout.addWidget(QLabel("Fixed:"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(1)
        self.interval_spin.setMaximum(100)
        self.interval_spin.setValue(4)  # Default to 4
        self.interval_spin.valueChanged.connect(self.update_spawn_interval)
        fixed_interval_layout.addWidget(self.interval_spin)
        interval_layout.addLayout(fixed_interval_layout)
        
        random_interval_layout = QHBoxLayout()
        self.random_interval_check = QCheckBox("Random")
        self.random_interval_check.setChecked(True)  # Default enabled
        self.random_interval_check.stateChanged.connect(self.toggle_random_interval)
        random_interval_layout.addWidget(self.random_interval_check)
        
        min_max_layout = QHBoxLayout()
        min_max_layout.addWidget(QLabel("Min:"))
        self.min_interval_spin = QSpinBox()
        self.min_interval_spin.setMinimum(1)
        self.min_interval_spin.setMaximum(50)
        self.min_interval_spin.setValue(4)  # Default to 4
        self.min_interval_spin.valueChanged.connect(self.update_min_interval)
        min_max_layout.addWidget(self.min_interval_spin)
        
        min_max_layout.addWidget(QLabel("Max:"))
        self.max_interval_spin = QSpinBox()
        self.max_interval_spin.setMinimum(1)
        self.max_interval_spin.setMaximum(100)
        self.max_interval_spin.setValue(15)
        self.max_interval_spin.valueChanged.connect(self.update_max_interval)
        min_max_layout.addWidget(self.max_interval_spin)
        random_interval_layout.addLayout(min_max_layout)
        interval_layout.addLayout(random_interval_layout)
        
        interval_group.setLayout(interval_layout)
        basic_layout.addWidget(interval_group)
        
        # Vehicles per spawn
        count_group = QGroupBox("Vehicles per Spawn")
        count_layout = QVBoxLayout()
        count_layout.setSpacing(2)
        
        fixed_count_layout = QHBoxLayout()
        fixed_count_layout.addWidget(QLabel("Fixed:"))
        self.count_spin = QSpinBox()
        self.count_spin.setMinimum(1)
        self.count_spin.setMaximum(10)
        self.count_spin.setValue(5)  # Default to 5
        self.count_spin.valueChanged.connect(self.update_spawn_count)
        fixed_count_layout.addWidget(self.count_spin)
        count_layout.addLayout(fixed_count_layout)
        
        random_count_layout = QHBoxLayout()
        self.random_count_check = QCheckBox("Random")
        self.random_count_check.setChecked(True)  # Default enabled
        self.random_count_check.stateChanged.connect(self.toggle_random_count)
        random_count_layout.addWidget(self.random_count_check)
        
        min_max_count_layout = QHBoxLayout()
        min_max_count_layout.addWidget(QLabel("Min:"))
        self.min_count_spin = QSpinBox()
        self.min_count_spin.setMinimum(1)
        self.min_count_spin.setMaximum(5)
        self.min_count_spin.setValue(1)
        self.min_count_spin.valueChanged.connect(self.update_min_count)
        min_max_count_layout.addWidget(self.min_count_spin)
        
        min_max_count_layout.addWidget(QLabel("Max:"))
        self.max_count_spin = QSpinBox()
        self.max_count_spin.setMinimum(1)
        self.max_count_spin.setMaximum(10)
        self.max_count_spin.setValue(6)  # Default to 6
        min_max_count_layout.addWidget(self.max_count_spin)
        random_count_layout.addLayout(min_max_count_layout)
        count_layout.addLayout(random_count_layout)
        
        count_group.setLayout(count_layout)
        basic_layout.addWidget(count_group)
        
        layout.addLayout(basic_layout)
        
        # Vehicle type distribution
        type_group = QGroupBox("Vehicle Type Distribution")
        type_layout = QVBoxLayout()
        type_layout.setSpacing(2)
        
        self.type_sliders = {}
        for vehicle_type, percentage in self.sim_thread.vehicle_types.items():
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(f"{vehicle_type}:"))
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(percentage)
            slider.valueChanged.connect(lambda v, t=vehicle_type: self.update_vehicle_type_distribution(t, v))
            self.type_sliders[vehicle_type] = slider
            slider_layout.addWidget(slider)
            
            label = QLabel(f"{percentage}%")
            slider_layout.addWidget(label)
            self.type_sliders[f"{vehicle_type}_label"] = label
            
            type_layout.addLayout(slider_layout)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Route distribution
        route_group = QGroupBox("Route Distribution")
        route_layout = QVBoxLayout()
        route_layout.setSpacing(2)
        
        self.route_sliders = {}
        for route, weight in self.sim_thread.route_weights.items():
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(f"{route}:"))
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(weight)
            slider.valueChanged.connect(lambda v, r=route: self.update_route_distribution(r, v))
            self.route_sliders[route] = slider
            slider_layout.addWidget(slider)
            
            label = QLabel(f"{weight}%")
            slider_layout.addWidget(label)
            self.route_sliders[f"{route}_label"] = label
            
            route_layout.addLayout(slider_layout)
        
        route_group.setLayout(route_layout)
        layout.addWidget(route_group)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
        # Set N-S Dominant as default after all UI elements are created
        self.preset_buttons[4].setChecked(True)  # Index 4 is N-S Dominant
        self.apply_distribution_preset(5)  # Apply N-S Dominant preset
    
    def create_vehicle_panel(self, parent_layout):
        group = QGroupBox("Add Vehicle")
        layout = QHBoxLayout()
        
        # Route selection
        self.route_combo = QComboBox()
        self.route_combo.addItems([
            "W_N", "W_E", "W_S",
            "N_W", "N_E", "N_S",
            "E_N", "E_S", "E_W",
            "S_N", "S_E", "S_W"
        ])
        layout.addWidget(QLabel("Route:"))
        layout.addWidget(self.route_combo)
        
        # Vehicle type selection
        self.vehicle_type_combo = QComboBox()
        self.vehicle_type_combo.addItems([
            "veh_passenger", "veh_bus", "veh_truck", "veh_emergency", "veh_motorcycle"
        ])
        layout.addWidget(QLabel("Type:"))
        layout.addWidget(self.vehicle_type_combo)
        
        # Speed input
        self.speed_input = QLineEdit("10")
        layout.addWidget(QLabel("Speed:"))
        layout.addWidget(self.speed_input)
        
        # Lane selection
        self.lane_combo = QComboBox()
        self.lane_combo.addItems(["random", "0", "1", "2", "3"])
        layout.addWidget(QLabel("Lane:"))
        layout.addWidget(self.lane_combo)
        
        # Add vehicle button
        add_button = QPushButton("Add Vehicle")
        add_button.clicked.connect(self.add_vehicle)
        layout.addWidget(add_button)
        
        # Add random vehicles button
        random_button = QPushButton("Add 5 Random")
        random_button.clicked.connect(self.add_random_vehicles)
        layout.addWidget(random_button)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_vehicle_table(self, parent_layout):
        group = QGroupBox("Active Vehicles")
        layout = QVBoxLayout()
        
        # Create table
        self.vehicle_table = QTableWidget()
        self.vehicle_table.setColumnCount(7)  # Added column for vehicle type
        self.vehicle_table.setHorizontalHeaderLabels(["ID", "Type", "Route", "Road", "Lane", "Speed", "Waiting"])
        layout.addWidget(self.vehicle_table)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_vehicle)
        button_layout.addWidget(remove_button)
        
        remove_all_button = QPushButton("Remove All")
        remove_all_button.clicked.connect(self.remove_all_vehicles)
        button_layout.addWidget(remove_all_button)
        
        highlight_button = QPushButton("Highlight Selected")
        highlight_button.clicked.connect(self.highlight_vehicle)
        button_layout.addWidget(highlight_button)
        
        layout.addLayout(button_layout)
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def create_statistics_panel(self, parent_layout):
        group = QGroupBox("Intersection Statistics")
        layout = QHBoxLayout()
        
        # Create statistics for each direction
        for direction in ["North", "South", "East", "West"]:
            direction_layout = QVBoxLayout()
            direction_layout.addWidget(QLabel(direction))
            
            count_label = QLabel("Count: 0")
            queue_label = QLabel("Queue: 0")
            speed_label = QLabel("Speed: 0 m/s")
            
            direction_layout.addWidget(count_label)
            direction_layout.addWidget(queue_label)
            direction_layout.addWidget(speed_label)
            
            setattr(self, f"{direction.lower()}_count", count_label)
            setattr(self, f"{direction.lower()}_queue", queue_label)
            setattr(self, f"{direction.lower()}_speed", speed_label)
            
            layout.addLayout(direction_layout)
        
        # Traffic light control
        light_layout = QVBoxLayout()
        light_layout.addWidget(QLabel("Traffic Light"))
        
        self.light_phase_label = QLabel("Phase: N/A")
        light_layout.addWidget(self.light_phase_label)
        
        self.manual_control = QCheckBox("Manual Control")
        light_layout.addWidget(self.manual_control)
        
        phase_layout = QHBoxLayout()
        for i, name in enumerate(["NS", "NSL", "EW", "EWL"]):
            button = QPushButton(name)
            button.clicked.connect(lambda checked, i=i: self.set_traffic_light_phase(i*2))
            phase_layout.addWidget(button)
        
        light_layout.addLayout(phase_layout)
        layout.addLayout(light_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def toggle_simulation(self):
        if not self.sim_thread.running:
            self.sim_thread.running = True
            self.sim_thread.start()
            self.start_button.setText("Stop Simulation")
            self.status_label.setText("Status: Running")
        else:
            self.sim_thread.running = False
            self.start_button.setText("Start Simulation")
            self.status_label.setText("Status: Stopped")
    
    def update_speed(self, value):
        self.sim_thread.speed = value
    
    def update_step(self, step):
        self.step_label.setText(f"Steps: {step}")
    
    def update_vehicles(self, vehicles):
        self.vehicle_table.setRowCount(len(vehicles))
        for row, (vid, data) in enumerate(vehicles.items()):
            self.vehicle_table.setItem(row, 0, QTableWidgetItem(vid))
            self.vehicle_table.setItem(row, 1, QTableWidgetItem(data.get('type', 'standard_car')))
            self.vehicle_table.setItem(row, 2, QTableWidgetItem(data['route']))
            self.vehicle_table.setItem(row, 3, QTableWidgetItem(data['road']))
            self.vehicle_table.setItem(row, 4, QTableWidgetItem(str(data['lane'])))
            self.vehicle_table.setItem(row, 5, QTableWidgetItem(str(data['speed'])))
            self.vehicle_table.setItem(row, 6, QTableWidgetItem(str(data['waiting'])))
    
    def update_statistics(self, stats):
        for direction in ["north", "south", "east", "west"]:
            if direction in stats:
                getattr(self, f"{direction}_count").setText(f"Count: {stats[direction]['count']}")
                getattr(self, f"{direction}_queue").setText(f"Queue: {stats[direction]['queue']}")
                getattr(self, f"{direction}_speed").setText(f"Speed: {stats[direction]['speed']:.1f} m/s")
        
        if 'light_phase' in stats:
            phase_names = ["NS Green", "NS Yellow", "NSL Green", "NSL Yellow", 
                          "EW Green", "EW Yellow", "EWL Green", "EWL Yellow"]
            phase = stats['light_phase']
            if 0 <= phase < len(phase_names):
                self.light_phase_label.setText(f"Phase: {phase_names[phase]}")
            else:
                self.light_phase_label.setText(f"Phase: {phase}")
    
    def add_vehicle(self):
        if not self.sim_thread.running:
            return
        
        try:
            route = self.route_combo.currentText()
            vehicle_type = self.vehicle_type_combo.currentText()
            speed = float(self.speed_input.text())
            lane = self.lane_combo.currentText()
            
            vehicle_id = f"{vehicle_type}_{route}_{self.vehicle_counter}"
            self.vehicle_counter += 1
        
            traci.vehicle.add(
                vehID=vehicle_id,
                routeID=route,
                typeID=vehicle_type,
                departLane=lane,
                departSpeed=str(speed)
            )
        except Exception as e:
            print(f"Error adding vehicle: {e}")
    
    def add_random_vehicles(self):
        if not self.sim_thread.running:
            return
        
        for _ in range(5):
            self.route_combo.setCurrentIndex(random.randint(0, self.route_combo.count()-1))
            self.vehicle_type_combo.setCurrentIndex(random.randint(0, self.vehicle_type_combo.count()-1))
            self.speed_input.setText(str(random.uniform(5, 15)))
            self.lane_combo.setCurrentIndex(random.randint(0, self.lane_combo.count()-1))
            self.add_vehicle()
    
    def remove_vehicle(self):
        if not self.sim_thread.running:
            return
        
        selected = self.vehicle_table.selectedItems()
        if selected:
            vehicle_id = self.vehicle_table.item(selected[0].row(), 0).text()
            try:
                traci.vehicle.remove(vehicle_id)
            except Exception as e:
                print(f"Error removing vehicle: {e}")
    
    def remove_all_vehicles(self):
        if not self.sim_thread.running:
            return
        
        try:
            for vid in traci.vehicle.getIDList():
                traci.vehicle.remove(vid)
        except Exception as e:
            print(f"Error removing vehicles: {e}")
    
    def highlight_vehicle(self):
        if not self.sim_thread.running:
            return
        
        selected = self.vehicle_table.selectedItems()
        if selected:
            vehicle_id = self.vehicle_table.item(selected[0].row(), 0).text()
            try:
                traci.vehicle.setColor(vehicle_id, (255, 255, 0, 255))
                traci.gui.trackVehicle("View #0", vehicle_id)
                traci.gui.setZoom("View #0", 3000)
            except Exception as e:
                print(f"Error highlighting vehicle: {e}")
    
    def set_traffic_light_phase(self, phase):
        if not self.sim_thread.running:
            return
        
        try:
            traci.trafficlight.setPhase("TL", phase)
        except Exception as e:
            print(f"Error setting traffic light phase: {e}")

    def toggle_auto_spawn(self, state):
        self.sim_thread.auto_spawn = bool(state)
    
    def update_spawn_interval(self, value):
        self.sim_thread.spawn_interval = value
    
    def update_spawn_count(self, value):
        self.sim_thread.spawn_count = value
    
    def toggle_random_interval(self, state):
        self.sim_thread.spawn_interval_random = bool(state)
        self.min_interval_spin.setEnabled(state)
        self.max_interval_spin.setEnabled(state)
    
    def toggle_random_count(self, state):
        self.sim_thread.spawn_count_random = bool(state)
        self.min_count_spin.setEnabled(state)
        self.max_count_spin.setEnabled(state)
    
    def update_min_interval(self, value):
        self.sim_thread.min_interval = value
        if value > self.max_interval_spin.value():
            self.max_interval_spin.setValue(value)
    
    def update_max_interval(self, value):
        self.sim_thread.max_interval = value
        if value < self.min_interval_spin.value():
            self.min_interval_spin.setValue(value)
    
    def update_min_count(self, value):
        self.sim_thread.min_count = value
        if value > self.max_count_spin.value():
            self.max_count_spin.setValue(value)
    
    def update_max_count(self, value):
        self.sim_thread.max_count = value
        if value < self.min_count_spin.value():
            self.min_count_spin.setValue(value)
    
    def update_vehicle_type_distribution(self, vehicle_type, value):
        self.sim_thread.vehicle_types[vehicle_type] = value
        self.type_sliders[f"{vehicle_type}_label"].setText(f"{value}%")
    
    def update_route_distribution(self, route, value):
        self.sim_thread.route_weights[route] = value
        self.route_sliders[f"{route}_label"].setText(f"{value}%")

    def toggle_render_mode(self, state):
        try:
            if state:
                # Switch to simple shapes
                view_file = os.path.join('intersection', 'view.xml')
                with open(view_file, 'w') as f:
                    f.write("""<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="standard"/>
    <delay value="20"/>
    <vehicleMode value="0"/>
    <vehicleQuality value="0"/>
    <vehicleName value="0"/>
    <vehicleSize value="1.0"/>
    <vehicleNameShow value="0"/>
    <vehicleNameSize value="50"/>
    <vehicleNameColor value="0,0,0"/>
    <vehicleNameBackground value="0"/>
    <vehicleNameBackgroundColor value="255,255,255"/>
    <vehicleNameBackgroundAlpha value="0.5"/>
    <vehicleNameBackgroundSize value="0.5"/>
    <vehicleNameBackgroundOffset value="0.0"/>
    <vehicleNameBackgroundRotation value="0.0"/>
    <vehicleNameBackgroundScale value="1.0"/>
    <minGap value="0.5"/>
</viewsettings>""")
            else:
                # Switch to real world rendering
                view_file = os.path.join('intersection', 'view.xml')
                with open(view_file, 'w') as f:
                    f.write("""<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="real world"/>
    <delay value="20"/>
    <vehicleMode value="9"/>
    <vehicleQuality value="3"/>
    <vehicleName value="0"/>
    <vehicleSize value="1.0"/>
    <vehicleNameShow value="0"/>
    <vehicleNameSize value="50"/>
    <vehicleNameColor value="0,0,0"/>
    <vehicleNameBackground value="0"/>
    <vehicleNameBackgroundColor value="255,255,255"/>
    <vehicleNameBackgroundAlpha value="0.5"/>
    <vehicleNameBackgroundSize value="0.5"/>
    <vehicleNameBackgroundOffset value="0.0"/>
    <vehicleNameBackgroundRotation value="0.0"/>
    <vehicleNameBackgroundScale value="1.0"/>
    <minGap value="2.5"/>
</viewsettings>""")
            
            print("Render mode toggled: ", state)
            
        except Exception as e:
            print(f"Error toggling render mode: {e}")

    def apply_distribution_preset(self, preset_num):
        if preset_num == 1:  # Urban Rush Hour
            # More passenger cars, some buses, few trucks
            self.sim_thread.vehicle_types = {
                "veh_passenger": 75,
                "veh_bus": 15,
                "veh_truck": 5,
                "veh_emergency": 3,
                "veh_motorcycle": 2
            }
            # More traffic on main roads
            self.sim_thread.route_weights = {
                "W_N": 12, "W_E": 15, "W_S": 8,
                "N_W": 8, "N_E": 15, "N_S": 12,
                "E_N": 15, "E_S": 8, "E_W": 12,
                "S_N": 8, "S_E": 12, "S_W": 15
            }
        elif preset_num == 2:  # Highway Traffic
            # More trucks, fewer passenger cars
            self.sim_thread.vehicle_types = {
                "veh_passenger": 45,
                "veh_bus": 10,
                "veh_truck": 35,
                "veh_emergency": 5,
                "veh_motorcycle": 5
            }
            # More through traffic
            self.sim_thread.route_weights = {
                "W_N": 10, "W_E": 20, "W_S": 10,
                "N_W": 10, "N_E": 20, "N_S": 10,
                "E_N": 10, "E_S": 20, "E_W": 10,
                "S_N": 10, "S_E": 20, "S_W": 10
            }
        elif preset_num == 3:  # Mixed Traffic
            # Even distribution of vehicle types
            self.sim_thread.vehicle_types = {
                "veh_passenger": 40,
                "veh_bus": 20,
                "veh_truck": 20,
                "veh_emergency": 10,
                "veh_motorcycle": 10
            }
            # Even distribution of routes
            self.sim_thread.route_weights = {
                "W_N": 8, "W_E": 8, "W_S": 8,
                "N_W": 8, "N_E": 8, "N_S": 8,
                "E_N": 8, "E_S": 8, "E_W": 8,
                "S_N": 8, "S_E": 8, "S_W": 8
            }
        elif preset_num == 4:  # Emergency Heavy
            # More emergency vehicles
            self.sim_thread.vehicle_types = {
                "veh_passenger": 30,
                "veh_bus": 10,
                "veh_truck": 10,
                "veh_emergency": 40,
                "veh_motorcycle": 10
            }
            # More traffic on emergency routes
            self.sim_thread.route_weights = {
                "W_N": 15, "W_E": 5, "W_S": 15,
                "N_W": 5, "N_E": 15, "N_S": 5,
                "E_N": 15, "E_S": 5, "E_W": 15,
                "S_N": 5, "S_E": 15, "S_W": 5
            }
        elif preset_num == 5:  # North-South Dominant
            # High passenger and motorcycle, low truck
            self.sim_thread.vehicle_types = {
                "veh_passenger": 45,
                "veh_bus": 10,
                "veh_truck": 2,
                "veh_emergency": 3,
                "veh_motorcycle": 40
            }
            # Heavy N-S traffic
            self.sim_thread.route_weights = {
                "W_N": 5, "W_E": 10, "W_S": 5,
                "N_W": 15, "N_E": 5, "N_S": 15,
                "E_N": 15, "E_S": 5, "E_W": 10,
                "S_N": 15, "S_E": 5, "S_W": 5
            }
        elif preset_num == 6:  # East-West Dominant
            # More passenger than motorcycle
            self.sim_thread.vehicle_types = {
                "veh_passenger": 55,
                "veh_bus": 12,
                "veh_truck": 1,
                "veh_emergency": 2,
                "veh_motorcycle": 30
            }
            # Heavy E-W traffic
            self.sim_thread.route_weights = {
                "W_N": 10, "W_E": 15, "W_S": 10,
                "N_W": 5, "N_E": 15, "N_S": 5,
                "E_N": 5, "E_S": 15, "E_W": 15,
                "S_N": 5, "S_E": 15, "S_W": 5
            }
        elif preset_num == 7:  # Diagonal Dominant
            # Equal passenger and motorcycle
            self.sim_thread.vehicle_types = {
                "veh_passenger": 40,
                "veh_bus": 8,
                "veh_truck": 2,
                "veh_emergency": 5,
                "veh_motorcycle": 45
            }
            # Heavy diagonal traffic
            self.sim_thread.route_weights = {
                "W_N": 15, "W_E": 5, "W_S": 5,
                "N_W": 5, "N_E": 15, "N_S": 5,
                "E_N": 5, "E_S": 15, "E_W": 5,
                "S_N": 5, "S_E": 5, "S_W": 15
            }
        elif preset_num == 8:  # Circular Flow
            # More motorcycle than passenger
            self.sim_thread.vehicle_types = {
                "veh_passenger": 35,
                "veh_bus": 7,
                "veh_truck": 1,
                "veh_emergency": 2,
                "veh_motorcycle": 55
            }
            # Circular traffic pattern
            self.sim_thread.route_weights = {
                "W_N": 15, "W_E": 5, "W_S": 5,
                "N_W": 5, "N_E": 15, "N_S": 5,
                "E_N": 5, "E_S": 15, "E_W": 5,
                "S_N": 5, "S_E": 5, "S_W": 15
            }
        
        # Update the UI sliders
        for vehicle_type, percentage in self.sim_thread.vehicle_types.items():
            self.type_sliders[vehicle_type].setValue(percentage)
            self.type_sliders[f"{vehicle_type}_label"].setText(f"{percentage}%")
        
        for route, weight in self.sim_thread.route_weights.items():
            self.route_sliders[route].setValue(weight)
            self.route_sliders[f"{route}_label"].setText(f"{weight}%")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())