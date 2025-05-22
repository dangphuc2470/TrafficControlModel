import os
import sys
import traci
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QLineEdit, QTableWidget, QTableWidgetItem, QGroupBox,
                            QCheckBox, QSlider)
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
        self.speed = 1.0
        
    def run(self):
        try:
            sumo_cmd = self.set_sumo(gui=True)
            traci.start(sumo_cmd)
            
            while self.running:
                step_duration = 1.0 / self.speed
                traci.simulationStep()
                self.step += 1
                
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
            "--delay", "100"
        ]
        
        return sumo_cmd
    
    def set_route_file(self):
        route_file = os.path.join("intersection", "interactive_routes.rou.xml")
        
        with open(route_file, "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Simulation Control")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create simulation thread
        self.sim_thread = SimulationThread()
        self.sim_thread.step_updated.connect(self.update_step)
        self.sim_thread.vehicle_updated.connect(self.update_vehicles)
        self.sim_thread.stats_updated.connect(self.update_statistics)
        
        # Create UI elements
        self.create_control_panel(layout)
        self.create_vehicle_panel(layout)
        self.create_vehicle_table(layout)
        self.create_statistics_panel(layout)
        
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
        
        # Status label
        self.status_label = QLabel("Status: Not running")
        layout.addWidget(self.status_label)
        
        # Step counter
        self.step_label = QLabel("Steps: 0")
        layout.addWidget(self.step_label)
        
        # Speed control
        layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self.update_speed)
        layout.addWidget(self.speed_slider)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
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
        self.vehicle_table.setColumnCount(6)
        self.vehicle_table.setHorizontalHeaderLabels(["ID", "Route", "Road", "Lane", "Speed", "Waiting"])
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
            self.vehicle_table.setItem(row, 1, QTableWidgetItem(data['route']))
            self.vehicle_table.setItem(row, 2, QTableWidgetItem(data['road']))
            self.vehicle_table.setItem(row, 3, QTableWidgetItem(str(data['lane'])))
            self.vehicle_table.setItem(row, 4, QTableWidgetItem(str(data['speed'])))
            self.vehicle_table.setItem(row, 5, QTableWidgetItem(str(data['waiting'])))
    
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
            speed = float(self.speed_input.text())
            lane = self.lane_combo.currentText()
            
            vehicle_id = f"{route}_{self.vehicle_counter}"
            self.vehicle_counter += 1
            
            traci.vehicle.add(
                vehID=vehicle_id,
                routeID=route,
                typeID="standard_car",
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())