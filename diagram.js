graph TD
    subgraph "External Dependencies"
TF["TensorFlow/Keras"]::: external
end

    subgraph "Environment"
SUMO["SUMO Simulator Environment"]::: sims
end

    subgraph "Traffic Generation"
TG["Traffic Generator"]::: tg
end

    subgraph "Simulation Controllers"
SimTrain["Simulation Controller (Training)"]::: sim
SimTest["Simulation Controller (Testing)"]::: sim
end

    subgraph "Deep Q-Learning Agent"
NN["Neural Network Model"]::: nn
Memory["Experience Replay (Memory)"]::: memory
end

    subgraph "Supporting Modules"
Viz["Visualization Module"]::: viz
Utils["Utility Functions"]::: utils
end


    %% Agent Service
    subgraph "Agent Service" 
        direction TB
Intersection["SUMO Config (intersection/)"]::: storage
AgentCommunicator["Agent Communicator"]::: api
TrainingMain["Training Orchestration"]::: compute
TestingMain["Testing Orchestration"]::: compute
ModelCore["RL/DL Core Modules"]::: compute
ModelStorage["Model Storage (models/)"]::: storage
end

    %% Central Server
    subgraph "Central Server"
        direction TB
FlaskAPI["Flask API"]::: api
DataStore["Data Store (JSON)"]::: storage
Figures["Figures Store"]::: storage
StaticAssets["Static Assets"]::: frontend
Templates["Templates"]::: frontend
end

    %% Frontend Dashboard
Dashboard["Frontend Dashboard"]::: frontend

    %% Data Flow Connections
TG-- >| "generates_routes" | SUMO
SUMO-- >| "state_update" | SimTrain
SUMO-- >| "state_update" | SimTest
SimTrain-- >| "state_to_agent" | NN
SimTest-- >| "state_to_agent" | NN
NN-- >| "action_command" | SimTrain
NN-- >| "action_command" | SimTest
SUMO-- >| "reward_feedback" | SimTrain
SUMO-- >| "reward_feedback" | SimTest
SimTrain-- >| "experience_capture" | Memory
SimTest-- >| "experience_capture" | Memory
Memory-- >| "experience_batch" | NN
NN-- >| "decision_loop" | Memory
SimTrain-- >| "metrics_data" | Viz
SimTest-- >| "metrics_data" | Viz
Utils-- >| "config_update" | NN
Utils-- >| "config_update" | SimTrain
Utils-- >| "config_update" | SimTest
TF-- >| "framework_usage" | NN
SUMO-- >| runs simulation | Intersection
Intersection-- >| provides network state | AgentCommunicator
AgentCommunicator-- >| HTTP POST / state | FlaskAPI
FlaskAPI-- >| store data | DataStore
FlaskAPI-- >| store figures | Figures
FlaskAPI-- >| HTTP GET / metrics | Dashboard
FlaskAPI-- >| HTTP GET / map data | Dashboard
Dashboard-- >| loads JS / CSS | StaticAssets
Dashboard-- >| renders views | Templates
FlaskAPI-- >| HTTP response with actions | AgentCommunicator

%% Training & Testing flows
TrainingMain-- >| uses communicator | AgentCommunicator
TrainingMain-- >| uses core modules | ModelCore
TrainingMain-- >| saves models | ModelStorage
TestingMain-- >| uses communicator | AgentCommunicator
TestingMain-- >| uses core modules | ModelCore
TestingMain-- >| reads models | ModelStorage


    %% Click Events
    click SUMO "https://github.com/nt505-p21-kltn/schedule-agent/tree/main/TLCS/intersection"
    click TG "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/generator.py"
    click NN "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/model.py"
    click NN "https://github.com/nt505-p21-kltn/schedule-agent/tree/main/TLCS/models"
    click Memory "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/memory.py"
    click SimTrain "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/training_main.py"
    click SimTrain "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/training_simulation.py"
    click SimTest "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/testing_main.py"
    click SimTest "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/testing_simulation.py"
    click Viz "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/visualization.py"
    click Utils "https://github.com/nt505-p21-kltn/schedule-agent/blob/main/TLCS/utils.py"

    %% Styles
    classDef external fill: #f9e79f, stroke: #D35400, stroke - width: 2px;
    classDef sims fill: #bbdef0, stroke:#0D47A1, stroke - width: 2px;
    classDef tg fill: #c8e6c9, stroke:#2E7D32, stroke - width: 2px;
    classDef sim fill: #fff9c4, stroke: #F57F17, stroke - width: 2px;
    classDef nn fill: #f8bbd0, stroke: #AD1457, stroke - width: 2px;
    classDef memory fill: #ffcdd2, stroke: #C62828, stroke - width: 2px;
    classDef viz fill: #b3e5fc, stroke:#0277BD, stroke - width: 2px;
    classDef utils fill: #ead7f2, stroke:#6A1B9A, stroke - width: 2px;
    classDef compute fill: #cfe2f3, stroke:#0b5394, stroke - width: 1px
    classDef storage fill: #d9ead3, stroke:#38761d, stroke - width: 1px
    classDef api fill: #ece0f8, stroke:#5b0f84, stroke - width: 1px
    classDef frontend fill: #fff2cc, stroke: #b45f06, stroke - width: 1px
    classDef external fill: #e7e6e6, stroke:#666666, stroke - width: 1px, stroke - dasharray: 5 5


