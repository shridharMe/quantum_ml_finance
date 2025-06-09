# Quantum ML for Financial Forecasting

This project demonstrates the integration of AI/ML with quantum technologies for financial time series analysis and forecasting.

## Project Structure

The project consists of the following files:

1. **requirements.txt** - Contains all necessary dependencies for the project
2. **README.md** - Documentation explaining the project purpose and structure
3. **quantum_finance_forecasting.py** - A standalone implementation of quantum ML for financial forecasting
4. **quantum_neural_network.py** - Detailed implementation of quantum neural networks
5. **financial_data_processor.py** - Module for acquiring and preprocessing financial data
6. **hybrid_quantum_model.py** - Implementation of a hybrid quantum-classical model
7. **main.py** - Main application script that demonstrates the complete workflow
8. **Dockerfile** - Multi-stage Docker configuration for containerized execution
9. **docker-compose.yml** - Docker Compose configuration for development
10. **docker-run.sh** - Helper script to build and run the Docker container

## Overview

This implementation showcases how quantum neural networks can be combined with classical machine learning techniques to potentially improve financial forecasting capabilities. The project uses PennyLane for quantum computing and TensorFlow for classical neural networks.

## Key Features

- **Quantum Neural Networks (QNNs)** that leverage quantum properties for financial data processing
- **Hybrid quantum-classical architecture** that combines quantum circuits with traditional neural networks
- **Financial time series forecasting** using historical stock market data
- **Technical indicator integration** for improved prediction capabilities
- **Visualization tools** for analyzing results and model performance

## Requirements

This project requires Python 3.8 or 3.9 (recommended).

Set up a virtual environment and install the required packages:

```bash
# Create a virtual environment
python -m venv quantum_ml_env

# Activate the virtual environment
# On macOS/Linux:
source quantum_ml_env/bin/activate
# On Windows:
# quantum_ml_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Note for macOS users with Apple Silicon (M1/M2/M3)

If you're using an Apple Silicon Mac, you may need to install the macOS-specific version of TensorFlow:

```bash
pip install tensorflow-macos
```

### Troubleshooting Installation Issues

If you encounter installation errors:

1. Make sure you're using Python 3.8 or 3.9
2. Try installing packages individually if the batch install fails
3. For TensorFlow issues on macOS, follow the [official TensorFlow macOS installation guide](https://developer.apple.com/metal/tensorflow-plugin/)

## Docker Support

This project includes Docker support for easy setup and execution regardless of your local Python version.

### Running with Docker

1. **Using the helper script:**
   ```bash
   ./docker-run.sh
   ```

2. **Manual Docker commands:**
   ```bash
   # Build the Docker image
   docker build -t quantum_ml_finance .
   
   # Run the application
   docker run -it --rm quantum_ml_finance
   
   # For interactive development shell
   docker run -it --rm quantum_ml_finance /bin/bash
   ```

3. **Using Docker Compose (for development):**
   ```bash
   # Start the container
   docker-compose up
   
   # Run in background
   docker-compose up -d
   
   # Access shell
   docker-compose exec quantum_ml bash
   ```

### Note on Docker Demo

The Docker demo uses a simplified model with a classical layer instead of a full quantum circuit to ensure compatibility across all environments. For the full quantum implementation, it's recommended to run the code directly in a properly configured Python environment.

## How It Works

The implementation follows these key steps:

1. **Data Collection**: Historical stock data is fetched using the yfinance library
2. **Data Preprocessing**: Time series data is scaled and formatted into sequences
3. **Technical Indicators**: Moving averages, RSI, MACD, and other indicators are added to enhance the feature set
4. **Quantum Feature Processing**: Input features are processed through a quantum circuit
5. **Hybrid Model**: A combined quantum-classical neural network makes predictions
6. **Evaluation**: Results are visualized and compared with actual values

## Quantum Circuit

The quantum circuit uses:
- RY gates for data encoding
- Parameterized RZ and RY gates for learning
- CNOT gates for entanglement
- PauliZ measurements for outputs

## Quantum Advantage

The quantum components potentially offer advantages through:
- Quantum superposition for exploring multiple financial scenarios simultaneously
- Quantum entanglement for capturing complex correlations in financial data
- Quantum interference for enhancing pattern recognition capabilities

## Running the Project

To run the project:
1. Ensure your virtual environment is activated
2. Execute the main script: `python main.py`

Alternatively, use Docker as described in the Docker Support section.

## Future Improvements

- Experiment with different quantum circuit architectures
- Implement quantum amplitude encoding for more efficient data representation
- Add more financial indicators as features
- Explore quantum advantage benchmarks against classical-only models
- Extend to other applications like weather forecasting, image processing for defect detection, or biological image processing for disease detection

## References

- [PennyLane Documentation](https://pennylane.ai/qml/)
- [Quantum Machine Learning for Finance](https://pennylane.ai/search?categories=quantum%20machine%20learning&sort=publication_date&contentType=DEMO)
- [TensorFlow Quantum](https://www.tensorflow.org/quantum)