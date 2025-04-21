from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS  # Thêm dòng này
import json
import os
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Thêm dòng này để cho phép truy cập từ Flutter Web

# Data storage
agent_data = {}
last_update = {}
TIMEOUT_THRESHOLD = 60  # seconds until agent considered offline

# Create directories for data storage
os.makedirs('server_data', exist_ok=True)
os.makedirs('server_data/figures', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

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
                print(f"Agent {agent_id} appears to be offline")
        
        # Generate visualizations if data exists
        if agent_data:
            try:
                generate_comparison_charts()
            except Exception as e:
                print(f"Error generating charts: {e}")
                
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

@app.route('/api/update', methods=['POST'])
def update_data():
    """Endpoint for agents to send their data"""
    try:
        data = request.json
        agent_id = data.get('agent_id')
        
        if not agent_id:
            return jsonify({'status': 'error', 'message': 'Missing agent_id'}), 400
        
        # Store the update time
        last_update[agent_id] = time.time()
        
        # Initialize agent data if it doesn't exist
        if agent_id not in agent_data:
            agent_data[agent_id] = {}
        
        # Update agent data
        for key, value in data.items():
            if key != 'agent_id':
                agent_data[agent_id][key] = value
        
        print(f"Received update from Agent {agent_id}")
        return jsonify({'status': 'success'}), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint to check the status of all agents"""
    current_time = time.time()
    status = {
        'agents': {},
        'total_agents': len(agent_data),
        'online_agents': 0
    }
    
    for agent_id in agent_data:
        is_online = agent_id in last_update and (current_time - last_update[agent_id] <= TIMEOUT_THRESHOLD)
        status['agents'][agent_id] = {
            'online': is_online,
            'last_update': last_update.get(agent_id, 0),
            'data_points': sum(len(value) if isinstance(value, list) else 1 for value in agent_data[agent_id].values()),
            'last_episode': agent_data[agent_id].get('last_episode', -1),
            'status': agent_data[agent_id].get('status', 'unknown')
        }
        if is_online:
            status['online_agents'] += 1
    
    return jsonify(status)

@app.route('/api/data', methods=['GET'])
def get_data():
    """Endpoint to retrieve all collected data"""
    return jsonify(agent_data)

@app.route('/api/latest_charts', methods=['GET'])
def get_latest_charts():
    """Get information about the latest generated charts"""
    charts = {
        'rewards_chart': '/static/rewards_comparison.png?t=' + str(int(time.time())),
        'queue_chart': '/static/queue_comparison.png?t=' + str(int(time.time())),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(charts)

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

    # Start the background thread for saving data
    bg_thread = threading.Thread(target=save_data_periodically, daemon=True)
    bg_thread.start()
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=False)