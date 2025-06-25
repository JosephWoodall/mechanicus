# mechanicus

This is my space to explore the wonderful world of brain-computer interfaces. The majority of the decision making here assumes that you have
an EEG Headset (or EEG data) handy. Brain-waves are the main data source I am leveraging in this project.

So, essentially, the flow is:
EEG Headset captures brain waves -> ML Model interprets them -> software executes hardware movement based on ML model's interpretation

Initial PoC Data offered for free here: https://www.physionet.org/content/eegmmidb/1.0.0/S001/#files-panel
You can also generate your own.

I plan to expand the binary classifier to multi-class classifier for 3D movement in a cartesian plane, where the output is a hash value coorelating to spherical coordinates; exercised by a lookup algorithm.

I am still brainstorming how to create the prosthetic apparatus. I might need help in this area.

# How to Use

Currently, the main way to run Mechanicus is by the following:

1. Activate your virtual environment, if you have one:
   Unix/Linux/MacOS:

```bash
source venv/bin/activate
```

Windows:

```bash
source venv/Scripts/activate
```

2. Then run the following

```bash
pip install requirements.txt
python main.py
```

# External Requirements

This project utilzes pyfirmata2, which requires the upload of StandardFirmata to the Arduino board. Please follow the tutorial in the URL below in order to get started before running any of this code:
https://github.com/berndporr/pyFirmata2?tab=readme-ov-file

# POC Logical Flow

```mermaid
flowchart TD
    A[main.py] --> B[Load YAML Config]
    B --> C[Check & Generate Files]
    C --> D[action.py]
    D --> E[HashToServoLookup]
    D --> F[ServoController]
    D --> G[ML Inference]
    G --> H[Hash to Servo Conversion]
    H --> I[Servo Movement]

    %% File Dependencies
    J[hash_to_servo_lookup.json] --> E
    K[inference_data.json] --> G
    L[inference_model.pkl] --> G

    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style I fill:#e8f5e8
```
# Desired End Result Logical Flow
```mermaid
flowchart TD
    %% End User Flow
    A[Device Powers On] --> B[Initialize Hardware & Load Models]
    B --> C[Monitor EEG Sensors]
    C --> D{EEG Anomaly Spike Detected?}
    D -->|No| C
    D -->|Yes| E[Collect EEG Data Burst]
    E --> F[Pass to On-board ML Model]
    F --> G[Model Predicts Target Position Hash]
    G --> H[Load Current Servo Positions]
    H --> I[RL Agent Plans Optimal Path]
    I --> J[Execute Servo Movement Sequence]
    J --> K[Update Current Position State]
    K --> C
    
    %% Offline Training Flow
    L[START: Offline Training Phase] --> M[Acquire Training Data]
    M --> N[Generate EEG-Position Datasets]
    N --> O[Create Sample Inference Data]
    
    O --> P{Parallel Training}
    P --> Q[Train ML Model<br/>EEG → Position Hash]
    P --> R[Train RL Agent<br/>Path Optimization]
    
    Q --> S[Save ML Model<br/>inference_model.pkl]
    R --> T[Save RL Agent<br/>rl_agent.pkl]
    
    S --> U[Model Validation & Testing]
    T --> U
    U --> V[Deploy Models to Device]
    
    %% New Version Deployment Flow
    W[START: New Version Deployment] --> X[Create Test Environment]
    X --> Y[Deploy New Models in Test Mode]
    Y --> Z[Run Mechanicus Test Pipeline]
    Z --> AA{All Tests Pass?}
    AA -->|No| AB[Debug & Fix Issues]
    AB --> Y
    AA -->|Yes| AC[Deploy to Production]
    AC --> AD[Update Device Models]
    AD --> AE[Restart Device with New Models]
    AE --> A
    
    %% Styling
    style A fill:#e1f5fe
    style L fill:#f3e5f5
    style W fill:#fff3e0
    style D fill:#ffebee
    style P fill:#e8f5e8
    style AA fill:#ffebee
    
    %% Subgraph groupings
    subgraph "End User Runtime"
        A
        B
        C
        D
        E
        F
        G
        H
        I
        J
        K
    end
    
    subgraph "Offline Training Pipeline"
        L
        M
        N
        O
        P
        Q
        R
        S
        T
        U
        V
    end
    
    subgraph "Deployment Pipeline"
        W
        X
        Y
        Z
        AA
        AB
        AC
        AD
        AE
    end
```
### Detailed Implementation Flow of Above
#### End User Runtime Flow
1. Device Powers On:
   - Load mechanicus_run_configuration.yaml
   - Initialize servo controllers
   - Load inference_model.pkl & rl_agent.pkl
   - Calibrate EEG sensors

2. EEG Monitoring Loop:
   - Continuous sensor monitoring
   - Anomaly detection algorithms
   - Trigger on significant spikes

3. Action Execution:
   - EEG → ML Model → Position Hash
   - Current Position + Target Position → RL Agent
   - RL Agent → Optimal movement sequence
   - Execute servo movements
#### Offline Training Pipeline
1. Training Data Acquisition:
   - generate_hash_lookup.py → hash_to_servo_lookup.json
   - data_collection.py → training_data.json + inference_data.json
   - Real EEG data collection (optional)

2. Parallel Training:
   ML Model Training:
      - Input: EEG data features
      - Output: Position hash predictions
      - Algorithm: Random Forest/Neural Network
      - Save: inference_model.pkl
   
   RL Agent Training:
      - State: Current servo positions
      - Action: Servo angle adjustments
      - Reward: Smooth movement + target achievement
      - Algorithm: Q-Learning/PPO
      - Save: rl_agent.pkl

3. Validation:
   - Cross-validation on test datasets
   - Performance metrics collection
   - Integration testing
#### Deployment Pipeline
1. Test Environment:
   - test_mode: true in configuration
   - Simulated hardware interactions
   - Controlled test scenarios

2. Test Pipeline:
   - Unit tests for each component
   - Integration tests for full pipeline
   - Performance benchmarking
   - Safety validation

3. Production Deployment:
   - Model versioning and rollback capability
   - Gradual rollout strategy
   - Monitoring and logging
   - Remote update capability

# Key Components Still Needed for Implementation
# 1. EEG Anomaly Detection
src/eeg_monitor.py:
    - Real-time EEG data collection
    - Anomaly spike detection
    - Data preprocessing for ML model

# 2. Reinforcement Learning Agent
src/rl_agent.py:
    - Path planning from current to target position
    - Servo movement optimization
    - Training environment simulation

# 3. Device Runtime Controller
src/device_runtime.py:
    - Main runtime loop
    - EEG monitoring integration
    - Model coordination
    - Hardware control

# 4. Training Orchestrator
src/train_rl.py:
    - RL agent training pipeline
    - Environment simulation
    - Reward function definition

# 5. Deployment Manager
src/deployment.py:
    - Model versioning
    - Test pipeline execution
    - Production deployment
