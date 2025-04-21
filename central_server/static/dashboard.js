// Global variables
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
}