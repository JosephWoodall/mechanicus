# Mechanicus Platform

The Mechanicus Platform is a modularized microservice architecture designed for EEG data processing, servo control, data collection, and machine learning model training. This project aims to provide a robust solution for real-time EEG monitoring and control through a user-facing chatbot interface and machine learning forecasting capabilities.

## Project Structure

```
mechanicus-platform
├── services
│   ├── eeg-processor         # Service for processing EEG data
│   ├── servo-controller       # Service for controlling servo movements
│   ├── data-collector         # Service for collecting and preparing data
│   └── training-pipeline      # Service for training machine learning models
├── shared                     # Shared resources across services
├── docker-compose.yml         # Docker Compose configuration for local development
├── docker-compose.prod.yml    # Docker Compose configuration for production
├── docker-compose.test.yml     # Docker Compose configuration for testing
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Docker and Docker Compose installed on your machine.
- Python 3.x for local development (if needed).

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/microsoft/vscode-remote-try-dab.git
   cd mechanicus-platform
   ```

2. Build the Docker images:
   ```
   docker-compose build
   ```

3. Start the services:
   ```
   docker-compose up
   ```

### Services Overview

- **EEG Processor**: Processes EEG data and runs inference using a machine learning model.
- **Servo Controller**: Controls servo movements based on processed EEG data.
- **Data Collector**: Collects EEG data and generates necessary datasets for training.
- **Training Pipeline**: Trains machine learning models and reinforcement learning agents.

### Usage

- Access the user-facing chatbot interface through the designated web endpoint.
- Monitor the logs of each service for real-time updates and debugging.

### Testing

To run tests, use the following command:
```
docker-compose -f docker-compose.test.yml up
```

### Deployment

For production deployment, use:
```
docker-compose -f docker-compose.prod.yml up
```

### Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

### License

This project is licensed under the MIT License. See the LICENSE file for details.