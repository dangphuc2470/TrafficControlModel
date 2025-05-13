import requests
import json
import time
import socket
import threading
import os
import numpy as np
import traci
import logging

logger = logging.getLogger(__name__)

class AgentCommunicatorTraining:
    def __init__(self, server_url, agent_id=None, mapping_config=None, env_file_path=None):
        """
        Initialize the communicator with the server URL
        
        Args:
            server_url: URL of the central server
            agent_id: Unique ID for this agent (if None, hostname will be used)
            location_data: Dictionary containing location information (lat, long, intersection name)
            env_file_path: Path to the environment.net.xml file
        """
        self.server_url = server_url
        self.agent_id = agent_id or socket.gethostname()
        self.data = {
            'agent_id': self.agent_id,
            'rewards': [],
            'queue_lengths': [],
            'waiting_times': [],
            'status': 'initializing',
            'last_episode': -1,
            'config': {},
            'model_info': {}
        }
        
        # Add new structure for current data (to be sent in next sync)
        self.current_data = {
            'agent_id': self.agent_id,
            'rewards': [],
            'queue_lengths': [],
            'waiting_times': [],
            'status': 'initializing',
            'last_episode': -1,
            'states': []
        }
        
        # Store location data separately - will be sent only on first sync
        self.mapping_config = mapping_config if mapping_config else {}
        self.env_file_path = env_file_path
        self.env_info = self._extract_env_info() if env_file_path else None
        self.topology_sent = False
        
        self.last_sync = 0
        self.sync_interval = 30  # seconds
        self.background_thread = None
        self.running = False
        
        # Create a directory to store data locally in case of connection issues
        self.backup_dir = f'agent_{self.agent_id}_data'
        os.makedirs(self.backup_dir, exist_ok=True)
        
        print(f"Agent communicator initialized with ID: {self.agent_id}")
        
        # Log what will be sent on first sync
        if self.mapping_config:
            print(f"Mapping configuration will be sent on first sync")
        if env_file_path:
            print(f"Environment topology data will be sent on first sync")
    
    def start_background_sync(self):
        """Start a background thread to periodically sync with the server"""
        if self.background_thread is not None and self.background_thread.is_alive():
            return  # Already running
        
        self.running = True
        self.background_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.background_thread.start()
        
    def stop_background_sync(self):
        """Stop the background sync thread"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
    
    def _sync_loop(self):
        """Background thread function to periodically sync with server"""
        while self.running:
            try:
                self.sync_with_server()
            except Exception as e:
                print(f"Error in background sync: {e}")
                # Backup data locally
                self._backup_data()
            
            # Sleep until next sync
            time.sleep(self.sync_interval)
    
    def _backup_data(self):
        """Save data locally as backup in case of server connection issues"""
        backup_file = os.path.join(self.backup_dir, f'backup_{int(time.time())}.json')
        with open(backup_file, 'w') as f:
            json.dump(self.data, f)
    
    def update_episode_result(self, episode, reward, queue_length, waiting_time=None):
        """Update results for a specific episode"""
        # Update both master data and current data
        self.data['last_episode'] = episode
        self.current_data['last_episode'] = episode
        
        self.data['rewards'].append(float(reward))
        self.current_data['rewards'].append(float(reward))
        
        self.data['queue_lengths'].append(float(queue_length))
        self.current_data['queue_lengths'].append(float(queue_length))
        
        if waiting_time is not None:
            self.data['waiting_times'].append(float(waiting_time))
            self.current_data['waiting_times'].append(float(waiting_time))
        
        # If it's been long enough since last sync, sync now
        current_time = time.time()
        if current_time - self.last_sync >= self.sync_interval:
            self.sync_with_server()
    
    def update_status(self, status):
        """Update the agent's status"""
        self.data['status'] = status
        
    def update_config(self, config):
        """Update the agent's configuration"""
        self.data['config'] = config
        
    def update_model_info(self, model_info):
        """Update information about the model"""
        self.data['model_info'] = model_info
    
    def sync_with_server(self):
        """Send only current data to the central server"""
        try:
            # Create a copy of only the current data to send
            send_data = self.current_data.copy()
            
            # Add topology data only on first sync
            if not self.topology_sent:
                topology_data = {}
                
                # Add mapping configuration if available
                if self.mapping_config:
                    topology_data.update(self.mapping_config)
                
                # Add environment data if available
                if self.env_info:
                    topology_data['environment'] = self.env_info
                
                # Only add topology section if we have data to send
                if topology_data:
                    send_data['topology'] = topology_data
            
            # Only send if there's actual data to send
            if (len(send_data['rewards']) > 0 or 
                len(send_data['queue_lengths']) > 0 or 
                len(send_data['states']) > 0 or 
                not self.topology_sent):
                
                print("Sending current data to server:", send_data)
                response = requests.post(
                    f"{self.server_url}/api/update", 
                    json=send_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"Successfully synced with server. Episodes: {len(self.current_data['rewards'])}")
                    self.last_sync = time.time()
                    
                    # Clear current data after successful sync
                    self.current_data['rewards'] = []
                    self.current_data['queue_lengths'] = []
                    self.current_data['waiting_times'] = []
                    self.current_data['states'] = []
                    
                    # Mark topology as sent if it was included
                    if not self.topology_sent and 'topology' in send_data:
                        self.topology_sent = True
                        print(f"Topology data sent to server for agent {self.agent_id}")
                    
                    return True
                else:
                    print(f"Server sync failed with status code: {response.status_code}")
                    return False
            else:
                print("No new data to send")
                return True
                
        except requests.exceptions.RequestException as e:
            print(f"Connection error during sync: {e}")
            return False
        
    def send_state(self, state, step, traffic_data=None):
        """
        Send the current state and traffic data to the central server
        
        Args:
            state: The state array representing cell occupancy
            step: Current simulation step
            traffic_data: Additional traffic information for coordination
        """
        if not 'states' in self.data:
            self.data['states'] = []
        
        # Get current traffic speeds from SUMO if available
        avg_speeds = {}
        try:
            traffic_speeds = {}
            for vehicle_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(vehicle_id)
                edge = traci.vehicle.getRoadID(vehicle_id)
                if edge not in traffic_speeds:
                    traffic_speeds[edge] = []
                traffic_speeds[edge].append(speed)
            
            # Calculate average speed for each edge
            for edge, speeds in traffic_speeds.items():
                if speeds:
                    avg_speeds[edge] = sum(speeds) / len(speeds)
            
            # Add average speeds to traffic data
            if traffic_data is None:
                traffic_data = {}
            traffic_data['avg_speed'] = avg_speeds
            
            logger.info(f"Calculated average speeds: {avg_speeds}")
        except Exception as e:
            logger.error(f"Warning: Could not get traffic speeds: {e}")
        
        state_data = {
            'step': step,
            'state': state,
            'timestamp': time.time(),
            'traffic_data': traffic_data or {},
            'speeds': avg_speeds  # Add speed data to state
        }
        
        # Add to both master data and current data
        self.data['states'].append(state_data)
        self.current_data['states'].append(state_data)
        
        # Limit the number of states we store to prevent memory issues
        if len(self.data['states']) > 100:  # Keep only the last 100 states
            self.data['states'] = self.data['states'][-100:]
        
        # If it's been long enough since last sync, sync now
        current_time = time.time()
        if current_time - self.last_sync >= self.sync_interval:
            self.sync_with_server()
    
    def get_coordination_data(self):
        """Get coordination data from the server"""
        try:
            response = requests.get(
                f"{self.server_url}/api/coordination/{self.agent_id}",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get coordination data: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting coordination data: {e}")
            return None

    def get_sync_timing(self):
        """Get synchronization timing data from the server"""
        try:
            response = requests.get(
                f"{self.server_url}/api/sync_times",
                timeout=5
            )
            if response.status_code == 200:
                sync_data = response.json()
                # Get timing data for this agent
                if self.agent_id in sync_data:
                    return sync_data[self.agent_id]
            return None
        except Exception as e:
            print(f"Error getting sync timing data: {e}")
            return None

    def _extract_env_info(self):
        """Extract relevant information from the environment.net.xml file"""
        if not self.env_file_path or not os.path.exists(self.env_file_path):
            print(f"Environment file not found: {self.env_file_path}")
            return None
            
        try:
            import xml.etree.ElementTree as ET
            
            # Parse the XML file
            tree = ET.parse(self.env_file_path)
            root = tree.getroot()
            
            # Extract location information
            location_elem = root.find('location')
            net_offset = location_elem.get('netOffset', '0.00,0.00') if location_elem else '0.00,0.00'
            conv_boundary = location_elem.get('convBoundary', '') if location_elem else ''
            
            # Extract junction information for the traffic light
            junctions = []
            for junction in root.findall('.//junction'):
                if junction.get('type') == 'traffic_light':
                    junctions.append({
                        'id': junction.get('id'),
                        'x': float(junction.get('x', 0)),
                        'y': float(junction.get('y', 0)),
                        'type': junction.get('type')
                    })
            
            # Extract edge information
            edges = []
            for edge in root.findall('.//edge'):
                if edge.get('function') != 'internal':  # Skip internal edges
                    edge_data = {
                        'id': edge.get('id'),
                        'from': edge.get('from', ''),
                        'to': edge.get('to', ''),
                        'lanes': []
                    }
                    
                    # Get lane information
                    for lane in edge.findall('lane'):
                        edge_data['lanes'].append({
                            'id': lane.get('id'),
                            'index': lane.get('index'),
                            'speed': lane.get('speed'),
                            'length': lane.get('length')
                        })
                    
                    edges.append(edge_data)
            
            # Extract traffic light phases
            tl_logic = []
            for tl in root.findall('.//tlLogic'):
                tl_data = {
                    'id': tl.get('id'),
                    'type': tl.get('type'),
                    'programID': tl.get('programID'),
                    'offset': tl.get('offset'),
                    'phases': []
                }
                
                for phase in tl.findall('phase'):
                    tl_data['phases'].append({
                        'duration': phase.get('duration'),
                        'state': phase.get('state')
                    })
                
                tl_logic.append(tl_data)
            
            return {
                'net_offset': net_offset,
                'boundary': conv_boundary,
                'junctions': junctions,
                'edges': edges,
                'tl_logic': tl_logic
            }
            
        except Exception as e:
            print(f"Error extracting environment information: {e}")
            return None