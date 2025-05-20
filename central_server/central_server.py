from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS 
import json
import os
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
import math
import xml.etree.ElementTree as ET
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from firebase_service import FirebaseService

app = Flask(__name__)
CORS(app)  # Cho phép truy cập từ Flutter Web

# Initialize Firebase service
firebase = FirebaseService()

# Global variables
active_agents = {}
agent_configs = {}
agent_metrics = {}

# Data storage
agent_data = {}
last_update = {}
TIMEOUT_THRESHOLD = 60  # seconds until agent considered offline

# Store recent server logs
server_logs = deque(maxlen=100)  # Keep the last 100 logs

# Create directories for data storage
os.makedirs('server_data', exist_ok=True)
os.makedirs('server_data/figures', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

def log_event(message):
    """Add a message to the server logs with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    server_logs.append(log_entry)
    print(log_entry)

def generate_intersection_map():
    """Generate a map showing all connected intersections with their network structure"""
    # Default center coordinates
    default_center = [10.777807, 106.681676]
    
    # Create a map centered on the specified location
    m = folium.Map(location=default_center, zoom_start=15)
    
    # Create a marker cluster for better visualization
    marker_cluster = MarkerCluster().add_to(m)
    
    # Track intersections with valid location data
    valid_intersections = {}
    
    # Add markers for each agent with location data
    has_markers = False
    for agent_id, data in agent_data.items():
        # Check if location data exists
        if 'topology' in data and 'location' in data['topology']:
            location = data['topology']['location']
            if 'latitude' in location and 'longitude' in location:
                try:
                    lat = float(location['latitude'])
                    lng = float(location['longitude'])
                    name = location.get('intersection_name', f'Intersection {agent_id}')
                    
                    # Determine agent status (online/offline)
                    is_online = agent_id in last_update and (time.time() - last_update[agent_id] <= TIMEOUT_THRESHOLD)
                    color = 'green' if is_online else 'red'
                    
                    # Get performance data if available
                    queue_info = ""
                    if 'queue_lengths' in data and len(data['queue_lengths']) > 0:
                        avg_queue = sum(data['queue_lengths'][-10:]) / min(10, len(data['queue_lengths']))
                        queue_info = f"<br>Average queue: {avg_queue:.2f} vehicles"
                    
                    # Create popup content with more detailed information
                    popup_content = f"""
                    <div style="width: 200px;">
                        <h3>{name}</h3>
                        <b>Agent ID:</b> {agent_id}<br>
                        <b>Status:</b> {'Online' if is_online else 'Offline'}{queue_info}<br>
                        <b>Location:</b> {lat:.6f}, {lng:.6f}
                    </div>
                    """
                    
                    # Add marker to the cluster
                    folium.Marker(
                        location=[lat, lng],
                        popup=folium.Popup(popup_content, max_width=250),
                        tooltip=name,
                        icon=folium.Icon(color=color, icon='traffic-light', prefix='fa')
                    ).add_to(marker_cluster)
                    
                    # Store the intersection for connection drawing
                    valid_intersections[agent_id] = {
                        'lat': lat,
                        'lng': lng,
                        'environment': data['topology'].get('environment', {}) if 'topology' in data else {},
                        'connected_to': data.get('connected_to', [])  # Get explicit connections
                    }
                    
                    has_markers = True
                    
                except (ValueError, TypeError) as e:
                    print(f"Error processing location for agent {agent_id}: {e}")
    
    # Draw connections between intersections based on explicit connections
    connections_drawn = set()
    for id1, info1 in valid_intersections.items():
        # Get list of connected intersections from config
        connected_to = info1.get('connected_to', [])
        if isinstance(connected_to, str):
            connected_to = [x.strip() for x in connected_to.split(',')]
            
        for id2 in connected_to:
            if id2 in valid_intersections and (id1, id2) not in connections_drawn and (id2, id1) not in connections_drawn:
                info2 = valid_intersections[id2]
                # Calculate distance for tooltip
                distance_km = haversine_distance(
                    (info1['lat'], info1['lng']), 
                    (info2['lat'], info2['lng'])
                )
                
                folium.PolyLine(
                    locations=[(info1['lat'], info1['lng']), (info2['lat'], info2['lng'])],
                    color='blue',
                    weight=2,
                    opacity=0.7,
                    tooltip=f"Distance: {distance_km:.2f} km"
                ).add_to(m)
                connections_drawn.add((id1, id2))
    
    # If no valid markers were added, add a default one
    if not has_markers:
        folium.Marker(
            location=default_center,
            popup="Default Location (No Agent Data)",
            tooltip="Default Location",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    # Save to static directory for serving
    map_path = 'static/intersection_map.html'
    m.save(map_path)
    
    # Also save as template
    with open('templates/map.html', 'w', encoding='utf-8') as f:  # Add encoding='utf-8'
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Control Network - Map View</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/static/style.css">
    <style>
        .map-container {
            width: 100%;
            height: calc(100vh - 200px);
            min-height: 600px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .dashboard-link {
            margin-bottom: 20px;
            display: inline-block;
            padding: 8px 16px;
            background: #f8f9fa;
            border-radius: 6px;
            text-decoration: none;
            color: #333;
        }
        .dashboard-link:hover {
            background: #e9ecef;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Traffic Light Control System - Network Map</h1>
            <div class="server-status">
                <span class="status-label">Server Status:</span>
                <span class="status-value online">Online</span>
            </div>
        </header>
        
        <a href="/" class="dashboard-link">← Back to Dashboard</a>
        
        <div class="map-container">
            <iframe src="/static/intersection_map.html" width="100%" height="100%" frameborder="0"></iframe>
        </div>
        
        <footer>
            <p>Traffic Light Control System - Central Server &copy; 2025</p>
        </footer>
    </div>
</body>
</html>
        ''')
    
    return map_path

def haversine_distance(point1, point2):
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

def calculate_intersection_sync_times():
    """Calculate sync times between connected intersections based on distance and expected travel time."""
    if not agent_data:
        log_event("No agent data available for calculating sync times")
        return {}
    
    sync_times = {}
    intersections_with_locations = {}
    
    # First, collect all intersections with valid location data
    for agent_id, data in agent_data.items():
        if 'topology' in data and 'location' in data['topology']:
            location = data['topology']['location']
            if 'latitude' in location and 'longitude' in location:
                try:
                    lat = float(location['latitude'])
                    lng = float(location['longitude'])
                    intersections_with_locations[agent_id] = (lat, lng)
                except (ValueError, TypeError):
                    log_event(f"Invalid location format for agent {agent_id}")
    
    # For each pair of intersections, calculate sync time
    for id1, loc1 in intersections_with_locations.items():
        sync_times[id1] = {}
        for id2, loc2 in intersections_with_locations.items():
            if id1 != id2:
                # Calculate distance between intersections
                distance_km = haversine_distance(loc1, loc2)
                
                # Calculate travel time based on average speed (assuming 40 km/h)
                avg_speed_kmh = 40.0
                
                # Get actual speed if available in agent data
                if id1 in agent_data and 'states' in agent_data[id1]:
                    states = agent_data[id1]['states']
                    if states and 'traffic_data' in states[-1] and 'avg_speed' in states[-1]['traffic_data']:
                        speeds = states[-1]['traffic_data']['avg_speed']
                        # Convert m/s to km/h and average all directions
                        if speeds:
                            avg_speed_kmh = sum(speeds.values()) * 3.6 / len(speeds)
                
                # Calculate travel time in seconds
                travel_time_sec = (distance_km / avg_speed_kmh) * 3600
                
                # Calculate optimal offset based on travel time (green wave)
                # This is a simple calculation - in real systems it would be more complex
                cycle_time = 0
                if id1 in agent_data and 'config' in agent_data[id1]:
                    green_duration = agent_data[id1]['config'].get('green_duration', 0)
                    yellow_duration = agent_data[id1]['config'].get('yellow_duration', 0)
                    cycle_time = (green_duration + yellow_duration) * 2  # Simplified cycle time calculation
                
                # Default cycle time if not available
                if cycle_time == 0:
                    cycle_time = 38  # Typical cycle time as fallback
                
                # Calculate optimal offset (modulo the cycle time)
                optimal_offset = travel_time_sec % cycle_time
                
                # Store the sync data
                sync_times[id1][id2] = {
                    "distance_km": round(distance_km, 2),
                    "travel_time_sec": round(travel_time_sec, 2),
                    "optimal_offset_sec": round(optimal_offset, 2),
                    "cycle_time_sec": cycle_time
                }
    
    # Save the sync times to a file for reference
    with open('server_data/sync_times.json', 'w') as f:
        json.dump(sync_times, f, indent=2)
    
    log_event(f"Calculated sync times between {len(intersections_with_locations)} intersections")
    for id1, targets in sync_times.items():
        for id2, sync_data in targets.items():
            log_event(f"  {id1} → {id2}: Distance={sync_data['distance_km']}km, "
                      
                     f"Travel Time={sync_data['travel_time_sec']}s, " +
                     f"Optimal Offset={sync_data['optimal_offset_sec']}s")
    return sync_times

def save_data_periodically():
    """Save collected data to disk periodically"""
    while True:
        # Save current data
        with open('server_data/agent_data.json', 'w') as f:
            json.dump(agent_data, f)
        
        # Check for disconnected agents
        current_time = time.time()
        for agent_id, last_time in list(last_update.items()):
            if current_time - last_time > TIMEOUT_THRESHOLD:
                log_event(f"WARNING: Agent {agent_id} appears to be offline")
        
        # Generate visualizations if data exists
        if agent_data:
            try:
                generate_comparison_charts()
                generate_intersection_map()
                log_event("Generated updated charts and map")
            except Exception as e:
                log_event(f"ERROR generating visualizations: {e}")
                
        time.sleep(30)  # Update every 30 seconds

def generate_comparison_charts():
    """Generate comparison charts from collected agent data"""
    if not agent_data:
        return
    
    # Prepare data for plotting
    agents = list(agent_data.keys())
    rewards = {agent: data.get('rewards', []) for agent, data in agent_data.items() if 'rewards' in data}
    queue_lengths = {agent: data.get('queue_lengths', []) for agent, data in agent_data.items() if 'queue_lengths' in data}
    
    # Only plot if we have data
    if rewards and any(len(r) > 0 for r in rewards.values()):
        # Plot rewards
        plt.figure(figsize=(12, 6))
        for agent, reward_data in rewards.items():
            if reward_data:
                plt.plot(reward_data, label=f"Agent {agent}")
        plt.title('Cumulative Rewards by Agent')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(f'server_data/figures/rewards_comparison.png')
        plt.savefig(f'static/rewards_comparison.png')
        plt.close()
    
    # Plot queue lengths if available
    if queue_lengths and any(len(q) > 0 for q in queue_lengths.values()):
        plt.figure(figsize=(12, 6))
        for agent, queue_data in queue_lengths.items():
            if queue_data:
                plt.plot(queue_data, label=f"Agent {agent}")
        plt.title('Average Queue Length by Agent')
        plt.xlabel('Episode')
        plt.ylabel('Queue Length')
        plt.legend()
        plt.savefig(f'server_data/figures/queue_comparison.png')
        plt.savefig(f'static/queue_comparison.png')
        plt.close()

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get overall system status"""
    status = firebase.get_system_status()
    return jsonify(status)

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get all agent data"""
    agents = firebase.get_all_agents()
    return jsonify(agents)

@app.route('/api/agent/<agent_id>', methods=['GET'])
def get_agent(agent_id):
    """Get specific agent data"""
    agent_data = firebase.get_agent_data(agent_id)
    if agent_data:
        return jsonify(agent_data)
    return jsonify({'error': 'Agent not found'}), 404

@app.route('/api/agent/<agent_id>/config', methods=['POST'])
def update_agent_config(agent_id):
    """Update agent configuration"""
    config = request.json
    firebase.update_agent_config(agent_id, config)
    return jsonify({'status': 'success'})

@app.route('/api/agent/<agent_id>/metrics', methods=['POST'])
def update_agent_metrics(agent_id):
    """Update agent metrics"""
    metrics = request.json
    firebase.update_agent_metrics(agent_id, metrics)
    return jsonify({'status': 'success'})

@app.route('/api/agent/<agent_id>/status', methods=['POST'])
def update_agent_status(agent_id):
    """Update agent status"""
    status = request.json.get('status')
    firebase.update_agent_status(agent_id, status)
    return jsonify({'status': 'success'})

@app.route('/api/agent/<agent_id>/location', methods=['POST'])
def update_agent_location(agent_id):
    """Update agent location"""
    data = request.json
    firebase.update_agent_location(
        agent_id,
        data.get('latitude'),
        data.get('longitude')
    )
    return jsonify({'status': 'success'})

@app.route('/api/agent/<agent_id>/performance', methods=['POST'])
def update_agent_performance(agent_id):
    """Update agent performance metrics"""
    performance_data = request.json
    firebase.update_agent_performance(agent_id, performance_data)
    return jsonify({'status': 'success'})

@app.route('/api/agent/<agent_id>/log', methods=['POST'])
def add_agent_log(agent_id):
    """Add agent log entry"""
    data = request.json
    firebase.add_agent_log(
        agent_id,
        data.get('message'),
        data.get('level', 'INFO')
    )
    return jsonify({'status': 'success'})

def update_system_status():
    """Periodically update system status"""
    while True:
        agents = firebase.get_all_agents()
        if agents:
            total_agents = len(agents)
            online_agents = sum(1 for agent in agents.values() 
                              if agent.get('status') == 'online')
            firebase.update_system_status(total_agents, online_agents)
        time.sleep(5)

@app.route('/api/latest_charts', methods=['GET'])
def get_latest_charts():
    """Get information about the latest generated charts"""
    charts = {
        'rewards_chart': '/static/rewards_comparison.png?t=' + str(int(time.time())),
        'queue_chart': '/static/queue_comparison.png?t=' + str(int(time.time())),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(charts)

@app.route('/api/reset', methods=['GET'])
def reset_server_data():
    """Clear all stored data and reset the server state"""
    global agent_data, last_update
    agent_data = {}
    last_update = {}
    
    print("Server data has been reset")
    return jsonify({'status': 'success', 'message': 'Server data has been reset'}), 200

@app.route('/map')
def show_map():
    """Serve the intersection map page"""
    return render_template('map.html')

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Endpoint to retrieve server logs"""
    return jsonify({'logs': list(server_logs)})

@app.route('/api/update', methods=['POST'])
def receive_updates():
    """Endpoint for agents to send their data"""
    try:
        data = request.json
        log_event(f"Received update from agent: {data['agent_id']}")
        agent_id = data.get('agent_id')
        
        if not agent_id:
            log_event("ERROR: Received update without agent_id")
            return jsonify({'status': 'error', 'message': 'Missing agent_id'}), 400
        
        # Store the update time
        last_update[agent_id] = time.time()
        
        # Initialize agent data if it doesn't exist
        if agent_id not in agent_data:
            agent_data[agent_id] = {
                'rewards': [],
                'queue_lengths': [],
                'waiting_times': [],
                'status': 'unknown',
                'online': True,  # New agents are online by default
                'last_episode': -1
            }
            log_event(f"New agent registered: {agent_id}")
        
        # Handle states data
        if 'states' in data:
            store_agent_states(agent_id, data['states'])
        
        # Handle one-time topology data
        if 'topology' in data and 'topology' not in agent_data[agent_id]:
            agent_data[agent_id]['topology'] = data['topology']
            try:
                generate_intersection_map()
            except Exception as e:
                log_event(f"Error generating map: {e}")
        
        # ACCUMULATE metrics data rather than replacing
        if 'rewards' in data and data['rewards']:
            agent_data[agent_id].setdefault('rewards', []).extend(data['rewards'])
            
        if 'queue_lengths' in data and data['queue_lengths']:
            agent_data[agent_id].setdefault('queue_lengths', []).extend(data['queue_lengths'])
            
        if 'waiting_times' in data and data['waiting_times']:
            agent_data[agent_id].setdefault('waiting_times', []).extend(data['waiting_times'])
        
        # Update scalar values
        if 'status' in data:
            agent_data[agent_id]['status'] = data['status']
            # Update online status based on the new status
            agent_data[agent_id]['online'] = data['status'] != 'terminated'
            # Update status in Firebase
            firebase.update_agent_status(agent_id, data['status'])
        
        # Update system status
        firebase.update_system_status(len(agent_data), sum(1 for agent in agent_data.values() if agent.get('online', False)))
        
        return jsonify({'status': 'success'})
    except Exception as e:
        log_event(f"ERROR in receive_updates: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
coordination_storage = {}  # Simple in-memory storage for coordination data

def store_agent_states(agent_id, states):
    """Store state data received from an agent"""
    if agent_id not in agent_data:
        agent_data[agent_id] = {}
    
    agent_data[agent_id]['states'] = states
    log_event(f"Stored {len(states)} states from agent {agent_id}")
    
    # Trigger coordination processing when new states arrive
    process_all_intersections()

def get_intersection_topology():
    """Get topology data for all intersections"""
    topology = {}
    for agent_id, data in agent_data.items():
        if 'topology' in data:
            topology[agent_id] = data['topology']
    return topology

def store_coordination_data(agent_id, data):
    """Store coordination data for an agent"""
    coordination_storage[agent_id] = {
        'data': data,
        'timestamp': time.time()
    }

def retrieve_coordination_data(agent_id):
    """Retrieve coordination data for an agent"""
    # Return hardcoded data if nothing specific is stored
    if (agent_id not in coordination_storage):
        # Hardcoded coordination data with default timing
        return {
            'recommended_phase': 0,  # NS_GREEN phase
            'duration_adjustment': 5,  # extend by 5 seconds
            'priority_direction': 'NS'  # prioritize north-south direction
        }
    
    return coordination_storage[agent_id]['data']

def process_all_intersections():
    """Process states from all intersections to generate coordination"""
    all_states = get_all_agent_states()
    if not all_states:
        return
    
    # Hardcoded coordination logic - alternating priorities based on agent ID
    for agent_id, states in all_states.items():
        if not states:
            continue
            
        latest_state = states[-1]  # Get most recent state
        traffic_data = latest_state.get('traffic_data', {})
        
        # Simple logic: if queue length is high in a direction, prioritize that direction
        queue_length = traffic_data.get('queue_length', 0)
        incoming = traffic_data.get('incoming_vehicles', {})
        
        # Determine direction with highest traffic
        ns_traffic = (incoming.get('N', 0) + incoming.get('S', 0))
        ew_traffic = (incoming.get('E', 0) + incoming.get('W', 0))
        
        if ns_traffic > ew_traffic:
            recommended_phase = 0  # NS_GREEN
            priority = 'NS'
        else:
            recommended_phase = 2  # EW_GREEN
            priority = 'EW'
            
        # Create coordination data
        coordination_data = {
            'recommended_phase': recommended_phase,
            'duration_adjustment': min(5, queue_length),  # Up to 5 seconds based on queue
            'priority_direction': priority
        }
        
        # Store the coordination data
        store_coordination_data(agent_id, coordination_data)
        log_event(f"Generated coordination for agent {agent_id}: priority={priority}")
        
def get_all_agent_states():
    """Retrieve states from all agents"""
    all_states = {}
    for agent_id, data in agent_data.items():
        if 'states' in data:
            all_states[agent_id] = data['states']
    return all_states

def calculate_travel_times(topology):
    """Calculate travel times between intersections based on topology"""
    travel_times = {}
    for agent_id, data in topology.items():
        if 'connections' in data:
            for connection in data['connections']:
                # Calculate travel time based on distance and speed limit
                distance = haversine_distance(
                    (data['location']['latitude'], data['location']['longitude']),
                    (connection['latitude'], connection['longitude'])
                )
                speed_limit = connection.get('speed_limit', 1)  # Default to 1 m/s to avoid division by zero
                travel_time = distance / speed_limit
                travel_times[(agent_id, connection['agent_id'])] = travel_time
    return travel_times

def get_drl_optimized_sync_times():
    """Get DRL-optimized synchronization times if available"""
    sync_file = 'server_data/sync_times.json'
    
    try:
        if os.path.exists(sync_file):
            with open(sync_file, 'r') as f:
                sync_times = json.load(f)
            
            # Check if this has DRL optimization flag
            has_drl = False
            for id1, targets in sync_times.items():
                for id2, data in targets.items():
                    if 'drl_optimized' in data and data['drl_optimized']:
                        has_drl = True
                        break
                if has_drl:
                    break
            
            if has_drl:
                log_event("Using DRL-optimized synchronization times")
                return sync_times
    except Exception as e:
        log_event(f"Error loading DRL sync times: {e}")
    
    # If not available or error, calculate them
    return calculate_intersection_sync_times()

@app.route('/api/sync_times', methods=['GET'])
def get_sync_times():
    """Get synchronization timing data"""
    try:
        sync_file = os.path.join(os.path.dirname(__file__), 'server_data', 'sync_times.json')
        if os.path.exists(sync_file):
            with open(sync_file, 'r') as f:
                sync_times = json.load(f)
            return jsonify(sync_times)
        else:
            return jsonify({}), 404
    except Exception as e:
        log_event(f"Error getting sync times: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create template files if they don't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Traffic Light Control System - Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/static/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <nav class="main-nav">
                <a href="/" class="nav-link active">Dashboard</a>
                <a href="/map" class="nav-link">Network Map</a>
            </nav>
            <h1>Traffic Light Control System - Central Server</h1>
            <div class="server-status">
                <span class="status-label">Server Status:</span>
                <span class="status-value online">Online</span>
                <span class="last-update">Last update: <span id="last-update-time">-</span></span>
            </div>
        </header>
        
        <div class="dashboard">
            <div class="sidebar">
                <div class="agent-summary">
                    <h2>Agent Summary</h2>
                    <div class="summary-stats">
                        <div class="stat-box">
                            <span class="stat-value" id="total-agents">0</span>
                            <span class="stat-label">Total Agents</span>
                        </div>
                        <div class="stat-box">
                            <span class="stat-value" id="online-agents">0</span>
                            <span class="stat-label">Online Agents</span>
                        </div>
                    </div>
                </div>
                
                <div class="agent-list">
                    <h2>Agent List</h2>
                    <div id="agent-list-container">
                        <p>No agents connected</p>
                    </div>
                </div>
            </div>
            
            <div class="main-content">
                <div class="chart-container">
                    <h2>Performance Comparison</h2>
                    <div class="chart-tabs">
                        <button class="tab-button active" onclick="showChart('rewards')">Rewards</button>
                        <button class="tab-button" onclick="showChart('queue')">Queue Length</button>
                    </div>
                    <div class="chart-display">
                        <div id="rewards-chart" class="chart active">
                            <img src="/static/rewards_comparison.png" alt="Rewards Chart" id="rewards-img">
                        </div>
                        <div id="queue-chart" class="chart">
                            <img src="/static/queue_comparison.png" alt="Queue Length Chart" id="queue-img">
                        </div>
                    </div>
                    <div class="chart-info">
                        <span>Last updated: <span id="chart-update-time">-</span></span>
                    </div>
                </div>
                    <div class="log-container">
                <h2>Server Log</h2>
                    <div class="log-controls">
                        <button id="refresh-logs" class="log-button">Refresh</button>
                        <button id="clear-logs" class="log-button">Clear Display</button>
                        <div class="auto-refresh">
                            <input type="checkbox" id="auto-refresh" checked>
                            <label for="auto-refresh">Auto-refresh</label>
                        </div>
                    </div>
                    <div class="log-box" id="log-box">
                        <div class="log-entry">Waiting for server logs...</div>
                    </div>
                </div>
                
                <div class="agent-details">
                    <h2>Agent Details</h2>
                    <select id="agent-selector">
                        <option value="">Select an agent</option>
                    </select>
                    <div id="agent-detail-container">
                        <p>Select an agent to view details</p>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Traffic Light Control System - Central Server &copy; 2025</p>
        </footer>
    </div>
    
    <script src="/static/dashboard.js"></script>
</body>
</html>''')

    # Create CSS file if it doesn't exist
    if not os.path.exists('static/style.css'):
        with open('static/style.css', 'w') as f:
            f.write('''* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background: #f4f6f9;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #fff;
    padding: 15px 25px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

header h1 {
    font-size: 1.8rem;
    color: #2c3e50;
}

.server-status {
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-label {
    font-weight: 600;
}

.status-value {
    padding: 5px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
}

.online {
    background: #d4edda;
    color: #155724;
}

.offline {
    background: #f8d7da;
    color: #721c24;
}

.dashboard {
    display: flex;
    gap: 20px;
}

.sidebar {
    width: 300px;
    flex-shrink: 0;
}

.agent-summary, .agent-list {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.summary-stats {
    display: flex;
    justify-content: space-between;
    margin-top: 15px;
}

.stat-box {
    width: 48%;
    padding: 15px;
    text-align: center;
    background: #f8f9fa;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 5px;
}

.stat-label {
    font-size: 0.9rem;
    color: #6c757d;
}

.agent-list h2, .agent-summary h2 {
    margin-bottom: 15px;
    font-size: 1.3rem;
    color: #2c3e50;
}

.agent-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 15px;
    border-bottom: 1px solid #eee;
}

.agent-item:last-child {
    border-bottom: none;
}

.agent-name {
    font-weight: 600;
}

.agent-status {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.agent-status.online {
    background: #28a745;
}

.agent-status.offline {
    background: #dc3545;
}

.main-content {
    flex-grow: 1;
}

.chart-container, .agent-details {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.chart-container h2, .agent-details h2 {
    margin-bottom: 15px;
    font-size: 1.3rem;
    color: #2c3e50;
}

.chart-tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
}

.tab-button {
    padding: 8px 15px;
    border: none;
    background: #f8f9fa;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
}

.tab-button.active {
    background: #007bff;
    color: white;
}

.chart-display {
    position: relative;
    height: 400px;
    border: 1px solid #eee;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 10px;
}

.chart {
    display: none;
    width: 100%;
    height: 100%;
}

.chart.active {
    display: block;
}

.chart img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.chart-info {
    text-align: right;
    color: #6c757d;
    font-size: 0.9rem;
}

select {
    width: 100%;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #ced4da;
    margin-bottom: 20px;
}

detail-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}

detail-item {
    padding: 15px;
    background: #f8f9fa;
    border-radius: 8px;
}

detail-label {
    font-weight: 600;
    margin-bottom: 8px;
    color: #6c757d;
}

detail-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #2c3e50;
}

footer {
    text-align: center;
    color: #6c757d;
    padding: 20px 0;
}

/* Agent status badges */
.status-badge {
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-badge.idle {
    background: #e2e3e5;
    color: #41464b;
}

.status-badge.training {
    background: #cff4fc;
    color: #055160;
}

.status-badge.simulating {
    background: #d1e7dd;
    color: #0f5132;
}

.status-badge.terminated {
    background: #f8d7da;
    color: #842029;
}
                    
.main-nav {
    display: flex;
    gap: 15px;
    margin-bottom: 20px;
}

.nav-link {
    padding: 8px 16px;
    background: #f8f9fa;
    border-radius: 6px;
    text-decoration: none;
    color: #333;
}

.nav-link:hover {
    background: #e9ecef;
}

.nav-link.active {
    background: #007bff;
    color: white;
}
                    
                    /* Log Box Styles */
.log-container {
    background: #fff;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.log-controls {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
}

.log-button {
    padding: 6px 12px;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
}

.log-button:hover {
    background: #e9ecef;
}

.auto-refresh {
    display: flex;
    align-items: center;
    margin-left: auto;
    font-size: 0.9rem;
}

.auto-refresh input {
    margin-right: 5px;
}

.log-box {
    height: 250px;
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 10px;
    overflow-y: auto;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.85rem;
    line-height: 1.4;
}

.log-entry {
    margin-bottom: 5px;
    padding-bottom: 5px;
    border-bottom: 1px solid #eee;
    word-break: break-word;
}

.log-entry:last-child {
    margin-bottom: 0;
    border-bottom: none;
}

.log-entry.error {
    color: #dc3545;
}

.log-entry.warning {
    color: #ffc107;
}

.log-entry.success {
    color: #28a745;
}

/* Responsive styles */
@media (max-width: 1000px) {
    .dashboard {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
    }
}''')

    # Create JavaScript file if it doesn't exist
    if not os.path.exists('static/dashboard.js'):
        with open('static/dashboard.js', 'w') as f:
            f.write('''// Global variables
let agentData = {};
let selectedAgent = null;
let refreshInterval = 5000; // 5 seconds

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initial data fetch
    fetchAgentStatus();
    fetchLatestCharts();
    
    // Set up periodic refreshes
    setInterval(fetchAgentStatus, refreshInterval);
    setInterval(fetchLatestCharts, refreshInterval * 6); // Refresh charts less frequently
    
    // Set up agent selector change event
    document.getElementById('agent-selector').addEventListener('change', function(e) {
        selectedAgent = e.target.value;
        updateAgentDetails();
    });
});

// Fetch the status of all agents
function fetchAgentStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            // Update UI with the received data
            updateStatusUI(data);
            // Also fetch detailed data
            fetchAgentData();
        })
        .catch(error => {
            console.error('Error fetching agent status:', error);
        });
}

// Fetch detailed data for all agents
function fetchAgentData() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            agentData = data;
            updateAgentDetails();
        })
        .catch(error => {
            console.error('Error fetching agent data:', error);
        });
}

// Fetch the latest chart information
function fetchLatestCharts() {
    fetch('/api/latest_charts')
        .then(response => response.json())
        .then(data => {
            document.getElementById('rewards-img').src = data.rewards_chart;
            document.getElementById('queue-img').src = data.queue_chart;
            document.getElementById('chart-update-time').textContent = data.timestamp;
        })
        .catch(error => {
            console.error('Error fetching latest charts:', error);
        });
}

// Update the UI with agent status information
function updateStatusUI(statusData) {
    // Update summary statistics
    document.getElementById('total-agents').textContent = statusData.total_agents;
    document.getElementById('online-agents').textContent = statusData.online_agents;
    document.getElementById('last-update-time').textContent = new Date().toLocaleTimeString();
    
    // Update agent list
    const agentListContainer = document.getElementById('agent-list-container');
    
    if (statusData.total_agents === 0) {
        agentListContainer.innerHTML = '<p>No agents connected</p>';
        return;
    }
    
    let agentListHTML = '';
    for (const [agentId, agentInfo] of Object.entries(statusData.agents)) {
        const statusClass = agentInfo.online ? 'online' : 'offline';
        const statusBadge = getStatusBadge(agentInfo.status);
        
        agentListHTML += `
            <div class="agent-item">
                <div>
                    <span class="agent-status ${statusClass}"></span>
                    <span class="agent-name">${agentId}</span>
                    ${statusBadge}
                </div>
                <div>
                    <span class="agent-episode">Ep: ${agentInfo.last_episode}</span>
                </div>
            </div>
        `;
    }
    
    agentListContainer.innerHTML = agentListHTML;
    
    // Update agent selector
    const agentSelector = document.getElementById('agent-selector');
    const currentValue = agentSelector.value;
    
    // Clear existing options except the first one
    while (agentSelector.options.length > 1) {
        agentSelector.remove(1);
    }
    
    // Add options for each agent
    for (const agentId of Object.keys(statusData.agents)) {
        const option = document.createElement('option');
        option.value = agentId;
        option.textContent = agentId;
        agentSelector.appendChild(option);
    }
    
    // Restore selected value if possible
    if (currentValue && Array.from(agentSelector.options).some(opt => opt.value === currentValue)) {
        agentSelector.value = currentValue;
    }
}

// Update the agent details section
function updateAgentDetails() {
    const detailContainer = document.getElementById('agent-detail-container');
    
    if (!selectedAgent || !agentData[selectedAgent]) {
        detailContainer.innerHTML = '<p>Select an agent to view details</p>';
        return;
    }
    
    const agent = agentData[selectedAgent];
    
    // Format configuration
    let configHTML = '';
    if (agent.config) {
        for (const [key, value] of Object.entries(agent.config)) {
            configHTML += `<div class="config-item">
                <span class="config-key">${key}:</span>
                <span class="config-value">${value}</span>
            </div>`;
        }
    }
    
    // Get last episode data
    const lastEpisode = agent.last_episode || 0;
    const rewardValue = agent.rewards && agent.rewards.length > 0 ? 
        agent.rewards[agent.rewards.length - 1].toFixed(2) : 'N/A';
    const queueValue = agent.queue_lengths && agent.queue_lengths.length > 0 ? 
        agent.queue_lengths[agent.queue_lengths.length - 1].toFixed(2) : 'N/A';
    
    detailContainer.innerHTML = `
        <div class="detail-grid">
            <div class="detail-item">
                <div class="detail-label">Status</div>
                <div class="detail-value">${agent.status || 'Unknown'}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Current Episode</div>
                <div class="detail-value">${lastEpisode}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Latest Reward</div>
                <div class="detail-value">${rewardValue}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Latest Queue Length</div>
                <div class="detail-value">${queueValue}</div>
            </div>
        </div>
        
        <h3 class="section-title">Agent Configuration</h3>
        <div class="config-container">
            ${configHTML || '<p>No configuration available</p>'}
        </div>
    `;
}

// Switch between chart tabs
function showChart(chartType) {
    // Update tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('active');
    });
    
    // Find the clicked button and activate it
    const activeButton = Array.from(tabButtons).find(button => {
        return button.textContent.toLowerCase().includes(chartType);
    });
    
    if (activeButton) {
        activeButton.classList.add('active');
    }
    
    // Update chart displays
    const charts = document.querySelectorAll('.chart');
    charts.forEach(chart => {
        chart.classList.remove('active');
    });
    
    document.getElementById(`${chartType}-chart`).classList.add('active');
}

// Helper function to get a status badge HTML
function getStatusBadge(status) {
    if (!status) return '';
    
    let badgeClass = 'idle';
    
    if (status === 'training') {
        badgeClass = 'training';
    } else if (status === 'simulating') {
        badgeClass = 'simulating';
    } else if (status === 'terminated') {
        badgeClass = 'terminated';
    }
    
    return `<span class="status-badge ${badgeClass}">${status}</span>`;
}''')

    # Generate an initial map
    try:
        generate_intersection_map()
        print("Generated initial map")
    except Exception as e:
        print(f"Error generating initial map: {e}")
    
    # Start the background thread for saving data
    bg_thread = threading.Thread(target=save_data_periodically, daemon=True)
    bg_thread.start()
    
    # Start system status update thread
    status_thread = threading.Thread(target=update_system_status)
    status_thread.daemon = True
    status_thread.start()
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=False)