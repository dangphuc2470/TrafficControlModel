import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from collections import deque
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sync_environment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SyncEnvironment")

class IntersectionSyncEnv(gym.Env):
    """
    Environment for training intersection synchronization.
    
    State: Traffic conditions and relative positioning between intersections
    Action: Offset adjustments between traffic signals
    Reward: Reduced waiting time and queue lengths across intersections
    """
    def __init__(self, intersection_data=None, max_offset=120):
        super(IntersectionSyncEnv, self).__init__()
        
        # Store intersection data that will be updated from agents
        self.intersection_data = intersection_data or {}
        self.intersection_ids = list(self.intersection_data.keys()) if self.intersection_data else []
        self.num_intersections = len(self.intersection_ids)
        
        # Max allowed offset in seconds
        self.max_offset = max_offset
        
        # Action space: Offset adjustments between each pair of connected intersections
        # For each pair, we define offset as percentage of cycle time (0-100%)
        if self.num_intersections > 1:
            self.action_space = spaces.Box(
                low=0.0, 
                high=1.0,  # Normalized offset (will be scaled to actual cycle time)
                shape=((self.num_intersections * (self.num_intersections - 1)) // 2,),
                dtype=np.float32
            )
        else:
            # Default action space for initialization when no intersections are available
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: Features for each intersection and pair of intersections
        # For each intersection: traffic volume, queue length, waiting time, cycle time
        # For each pair: distance, travel time, current offset
        features_per_intersection = 4  # traffic volume, queue length, waiting time, cycle time
        features_per_pair = 3  # distance, travel time, current offset
        
        if self.num_intersections > 0:
            total_features = (self.num_intersections * features_per_intersection + 
                             (self.num_intersections * (self.num_intersections - 1)) // 2 * features_per_pair)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
            )
        else:
            # Default observation space for initialization
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # Keep history of metrics for reward calculation
        self.history = {
            'waiting_times': deque(maxlen=10),
            'queue_lengths': deque(maxlen=10)
        }
        
        # Current state of the environment
        self.current_offsets = {}  # {(id1, id2): offset}
        self.distances = {}        # {(id1, id2): distance in km}
        self.travel_times = {}     # {(id1, id2): travel time in sec}
        self.cycle_times = {}      # {id: cycle time in sec}
    
    def update_intersection_data(self, new_data):
        """Update intersection data with new data from agents"""
        self.intersection_data = new_data
        self.intersection_ids = list(self.intersection_data.keys())
        self.num_intersections = len(self.intersection_ids)
        
        # Recalculate distances and travel times
        self._calculate_spatial_relationships()
        
        # Update action and observation spaces
        self._update_spaces()
    
    def _update_spaces(self):
        """Update action and observation spaces based on current intersections"""
        # Always use a fixed action space size based on maximum possible intersections
        max_intersections = 10  # Maximum number of intersections we want to support
        action_dim = (max_intersections * (max_intersections - 1)) // 2
        
        # Action space: offset adjustments for each pair
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Update observation space
        features_per_intersection = 4
        features_per_pair = 3
        total_features = (max_intersections * features_per_intersection + 
                         (max_intersections * (max_intersections - 1)) // 2 * features_per_pair)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
    
    def _calculate_spatial_relationships(self):
        """Calculate distances and travel times between intersections"""
        self.distances = {}
        self.travel_times = {}
        self.cycle_times = {}
        
        # Calculate cycle time for each intersection
        for id, data in self.intersection_data.items():
            if 'config' in data:
                green_duration = data['config'].get('green_duration', 30)
                yellow_duration = data['config'].get('yellow_duration', 4)
                cycle_time = (green_duration + yellow_duration) * 2  # Simplified cycle time calculation
            else:
                cycle_time = 38  # Default cycle time if not available
            self.cycle_times[id] = cycle_time
        
        # Calculate distances and travel times between intersections
        for i, id1 in enumerate(self.intersection_ids):
            for j in range(i+1, len(self.intersection_ids)):
                id2 = self.intersection_ids[j]
                
                # Get location data
                if ('topology' in self.intersection_data[id1] and 
                    'location' in self.intersection_data[id1]['topology'] and
                    'topology' in self.intersection_data[id2] and 
                    'location' in self.intersection_data[id2]['topology']):
                    
                    loc1 = self.intersection_data[id1]['topology']['location']
                    loc2 = self.intersection_data[id2]['topology']['location']
                    
                    try:
                        lat1 = float(loc1['latitude'])
                        lng1 = float(loc1['longitude'])
                        lat2 = float(loc2['latitude'])
                        lng2 = float(loc2['longitude'])
                        
                        # Calculate distance
                        distance_km = self._haversine_distance((lat1, lng1), (lat2, lng2))
                        self.distances[(id1, id2)] = distance_km
                        
                        # Calculate travel time (average vehicle speed assumption)
                        avg_speed_kmh = 40.0  # Default average speed
                        speed_source = "default"
                        
                        # Get actual speed if available
                        if ('states' in self.intersection_data[id1] and 
                            self.intersection_data[id1]['states']):
                            states = self.intersection_data[id1]['states']
                            if ('traffic_data' in states[-1] and 
                                'avg_speed' in states[-1]['traffic_data']):
                                speeds = states[-1]['traffic_data']['avg_speed']
                                # Convert m/s to km/h and average all directions
                                if speeds:
                                    avg_speed_kmh = sum(speeds.values()) * 3.6 / len(speeds)
                                    # Ensure minimum speed to prevent division by zero
                                    avg_speed_kmh = max(avg_speed_kmh, 5.0)  # Minimum 5 km/h
                                    speed_source = "realtime"
                                    logger.info(f"Using real-time speed data for {id1}-{id2}: {avg_speed_kmh:.2f} km/h")
                                else:
                                    logger.warning(f"No speed values available for {id1}-{id2}, using default speed")
                            else:
                                logger.warning(f"No traffic data available for {id1}-{id2}, using default speed")
                        else:
                            logger.warning(f"No states data available for {id1}-{id2}, using default speed")
                        
                        # Calculate travel time in seconds
                        travel_time_sec = (distance_km / avg_speed_kmh) * 3600
                        self.travel_times[(id1, id2)] = travel_time_sec
                        
                        logger.info(f"Travel time calculation for {id1}-{id2}: "
                                  f"distance={distance_km:.2f}km, "
                                  f"speed={avg_speed_kmh:.2f}km/h ({speed_source}), "
                                  f"time={travel_time_sec:.2f}s")
                        
                        # Initialize offset if not already set
                        if (id1, id2) not in self.current_offsets:
                            # Default offset is travel time modulo cycle time
                            cycle_time = min(self.cycle_times[id1], self.cycle_times[id2])
                            self.current_offsets[(id1, id2)] = travel_time_sec % cycle_time
                        
                    except (ValueError, KeyError, ZeroDivisionError) as e:
                        logger.error(f"Error calculating distance between {id1} and {id2}: {e}")
                        # Set default values on error
                        self.distances[(id1, id2)] = 0.1  # Small default distance
                        self.travel_times[(id1, id2)] = 30  # Default travel time
                        cycle_time = min(self.cycle_times.get(id1, 38), self.cycle_times.get(id2, 38))
                        self.current_offsets[(id1, id2)] = 0  # Default offset
    
    def _haversine_distance(self, point1, point2):
        """Calculate the great-circle distance between two points in kilometers"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r
    
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state"""
        super().reset(seed=seed)
        
        # Clear history
        self.history['waiting_times'].clear()
        self.history['queue_lengths'].clear()
        
        # Recalculate spatial relationships to ensure fresh start
        self._calculate_spatial_relationships()
        
        # Get initial state
        state = self._get_state()
        
        return state, {}
    
    def step(self, action):
        """
        Take an action in the environment
        
        Args:
            action: Normalized offsets for each pair of intersections
        
        Returns:
            state, reward, terminated, truncated, info
        """
        # Apply the offsets from the action
        self._apply_offsets(action)
        
        # Simulate effect of the action by collecting new metrics
        metrics = self._collect_current_metrics()
        
        # Store metrics in history
        self.history['waiting_times'].append(metrics['avg_waiting_time'])
        self.history['queue_lengths'].append(metrics['avg_queue_length'])
        
        # Calculate reward
        reward = self._calculate_reward(metrics)
        
        # Get the new state
        new_state = self._get_state()
        
        # This environment never terminates naturally
        terminated = False
        truncated = False
        
        # Return info dict with metrics
        info = {
            'metrics': metrics,
            'offsets': self.current_offsets.copy()
        }
        
        return new_state, reward, terminated, truncated, info
    
    def _apply_offsets(self, action):
        """Apply offset adjustments from the action"""
        idx = 0
        for i, id1 in enumerate(self.intersection_ids):
            for j in range(i+1, len(self.intersection_ids)):
                id2 = self.intersection_ids[j]
                if (id1, id2) in self.distances:  # Only apply to pairs with known distances
                    # Get the cycle time for this pair
                    cycle_time = min(self.cycle_times[id1], self.cycle_times[id2])
                    
                    # Convert normalized action to seconds
                    if idx < len(action):
                        self.current_offsets[(id1, id2)] = action[idx] * cycle_time
                        idx += 1
    
    def _collect_current_metrics(self):
        """Collect current performance metrics from all intersections"""
        total_waiting_time = 0
        total_queue_length = 0
        count = 0
        
        for id, data in self.intersection_data.items():
            # Get waiting time if available
            if 'waiting_times' in data and data['waiting_times']:
                total_waiting_time += data['waiting_times'][-1]
            
            # Get queue length if available
            if 'queue_lengths' in data and data['queue_lengths']:
                total_queue_length += data['queue_lengths'][-1]
            
            count += 1
        
        if count > 0:
            avg_waiting_time = total_waiting_time / count
            avg_queue_length = total_queue_length / count
        else:
            avg_waiting_time = 0
            avg_queue_length = 0
        
        return {
            'avg_waiting_time': avg_waiting_time,
            'avg_queue_length': avg_queue_length,
            'total_waiting_time': total_waiting_time,
            'total_queue_length': total_queue_length,
            'count': count
        }
    
    def _calculate_reward(self, current_metrics):
        """Calculate reward based on improvement in metrics"""
        reward = 0
        
        # If we have history, compare with previous metrics
        if len(self.history['waiting_times']) > 1:
            # Calculate improvement in waiting time
            prev_waiting = self.history['waiting_times'][-2]
            curr_waiting = current_metrics['avg_waiting_time']
            waiting_improvement = prev_waiting - curr_waiting
            
            # Calculate improvement in queue length
            prev_queue = self.history['queue_lengths'][-2]
            curr_queue = current_metrics['avg_queue_length']
            queue_improvement = prev_queue - curr_queue
            
            # Combine improvements as reward
            # Give more weight to waiting time improvement
            reward = 0.7 * waiting_improvement + 0.3 * queue_improvement
        
        return reward
    
    def _get_state(self):
        """Get the current state representation"""
        if not self.intersection_ids:
            # Return zero vector if no intersections
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        state_components = []
        
        # Add intersection features
        for id in self.intersection_ids:
            data = self.intersection_data[id]
            
            # Traffic volume (sum of incoming vehicles)
            traffic_volume = 0
            if ('states' in data and data['states'] and 
                'traffic_data' in data['states'][-1] and 
                'incoming_vehicles' in data['states'][-1]['traffic_data']):
                traffic_volume = sum(data['states'][-1]['traffic_data']['incoming_vehicles'].values())
            
            # Queue length
            queue_length = 0
            if 'queue_lengths' in data and data['queue_lengths']:
                queue_length = data['queue_lengths'][-1]
            
            # Waiting time
            waiting_time = 0
            if 'waiting_times' in data and data['waiting_times']:
                waiting_time = data['waiting_times'][-1]
            
            # Cycle time
            cycle_time = self.cycle_times.get(id, 38)
            
            # Add to state
            state_components.extend([traffic_volume, queue_length, waiting_time, cycle_time])
        
        # Add pair features
        for i, id1 in enumerate(self.intersection_ids):
            for j in range(i+1, len(self.intersection_ids)):
                id2 = self.intersection_ids[j]
                if (id1, id2) in self.distances:
                    # Distance
                    distance = self.distances[(id1, id2)]
                    
                    # Travel time
                    travel_time = self.travel_times[(id1, id2)]
                    
                    # Current offset
                    offset = self.current_offsets[(id1, id2)]
                    
                    # Add to state
                    state_components.extend([distance, travel_time, offset])
                else:
                    # Default values for unknown pairs
                    state_components.extend([0, 0, 0])
        
        # Convert to numpy array and normalize
        state = np.array(state_components, dtype=np.float32)
        
        # If state dimension doesn't match observation space, pad with zeros
        if state.shape[0] < self.observation_space.shape[0]:
            padding = np.zeros(self.observation_space.shape[0] - state.shape[0], dtype=np.float32)
            state = np.concatenate([state, padding])
        
        return state
    
    def get_optimal_offsets(self):
        """Return the current optimal offsets for all intersection pairs"""
        return self.current_offsets.copy()

    def _get_average_speed(self, agent1, agent2):
        """Get the average speed between two intersections based on their states"""
        try:
            # Get the latest states for both agents
            states1 = self.intersection_data[agent1].get('states', [])
            states2 = self.intersection_data[agent2].get('states', [])
            
            if not states1 or not states2:
                logger.warning(f"No states data available for {agent1} or {agent2}")
                return 40.0  # Default speed if no data available
            
            # Get the most recent state
            latest_state1 = states1[-1]
            latest_state2 = states2[-1]
            
            # Get speed data from traffic_data
            speeds1 = latest_state1.get('traffic_data', {}).get('avg_speed', {})
            speeds2 = latest_state2.get('traffic_data', {}).get('avg_speed', {})
            
            # Log speed data for debugging
            logger.info(f"Agent {agent1} speeds: {speeds1}")
            logger.info(f"Agent {agent2} speeds: {speeds2}")
            
            # Combine speeds from both intersections
            all_speeds = []
            all_speeds.extend(speeds1.values())
            all_speeds.extend(speeds2.values())
            
            if all_speeds:
                avg_speed = sum(all_speeds) / len(all_speeds)
                logger.info(f"Calculated average speed: {avg_speed:.2f} km/h")
                return avg_speed
            else:
                logger.warning(f"No speed data available for {agent1} or {agent2}")
                return 40.0  # Default speed if no data available
                
        except Exception as e:
            logger.error(f"Error calculating average speed: {e}")
            return 40.0  # Default speed on error