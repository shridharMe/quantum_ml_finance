"""
Quantum ML Financial Forecasting

This module demonstrates how to integrate quantum computing with machine learning
for financial time series forecasting using PennyLane and TensorFlow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pennylane as qml
from pennylane import numpy as qnp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define a quantum neural network (QNN) layer using PennyLane
n_qubits = 4  # Number of qubits
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    """
    Quantum circuit for feature processing.
    
    Args:
        inputs: Input features to encode into the quantum circuit
        weights: Trainable weights for the quantum circuit
    
    Returns:
        Expectation values from the quantum circuit
    """
    # Encode the classical input features into quantum states
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Apply trainable rotation gates with weights
    for i in range(n_qubits):
        qml.RZ(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
    
    # Apply entangling gates
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Measure expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer:
    """
    Quantum layer that can be integrated with classical neural networks.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.weights = np.random.uniform(0, 2*np.pi, (n_qubits, 2))
        
    def __call__(self, inputs):
        """Process inputs through the quantum circuit"""
        # Ensure inputs are properly scaled for the quantum circuit
        inputs_scaled = np.array(inputs) * np.pi  # Scale inputs to [0, π]
        
        # If inputs are batched, process each example separately
        if len(inputs_scaled.shape) > 1:
            return np.array([self._process_input(x) for x in inputs_scaled])
        return self._process_input(inputs_scaled)
    
    def _process_input(self, x):
        """Process a single input through the quantum circuit"""
        # Pad or truncate input to match n_qubits
        if len(x) < self.n_qubits:
            x = np.pad(x, (0, self.n_qubits - len(x)))
        else:
            x = x[:self.n_qubits]
        return quantum_circuit(x, self.weights)

def fetch_stock_data(ticker="AAPL", period="2y"):
    """
    Fetch historical stock data using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period to fetch
    
    Returns:
        DataFrame with stock data
    """
    stock_data = yf.download(ticker, period=period)
    return stock_data

def prepare_data(data, feature_columns=['Open', 'High', 'Low', 'Volume'], 
                target_column='Close', sequence_length=10, test_size=0.2):
    """
    Prepare time series data for training.
    
    Args:
        data: DataFrame with financial data
        feature_columns: Columns to use as features
        target_column: Column to predict
        sequence_length: Number of time steps to use for prediction
        test_size: Proportion of data to use for testing
    
    Returns:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y
    """
    # Extract features and target
    features = data[feature_columns].values
    target = data[target_column].values.reshape(-1, 1)
    
    # Scale the data
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i+sequence_length])
        y.append(target_scaled[i+sequence_length])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def build_hybrid_model(input_shape, quantum_layer):
    """
    Build a hybrid quantum-classical neural network.
    
    Args:
        input_shape: Shape of input data
        quantum_layer: Quantum layer instance
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Classical pre-processing layer
    model.add(Dense(16, activation='relu', input_shape=input_shape))
    model.add(Dense(n_qubits, activation='tanh'))  # Output matches quantum layer input
    
    # Custom quantum layer (wrapped in a Lambda layer)
    model.add(tf.keras.layers.Lambda(
        lambda x: quantum_layer(x),
        output_shape=(n_qubits,)
    ))
    
    # Classical post-processing layers
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    
    return model

def plot_predictions(actual, predicted, title="Stock Price Prediction"):
    """
    Plot actual vs predicted values.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_results.png')
    plt.show()

def main():
    # Fetch stock data
    print("Fetching stock data...")
    data = fetch_stock_data(ticker="AAPL", period="2y")
    print(f"Data shape: {data.shape}")
    
    # Prepare data
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_data(
        data, sequence_length=sequence_length
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Flatten the input for the model
    input_shape = (X_train.shape[1] * X_train.shape[2],)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Create quantum layer
    quantum_layer = QuantumLayer(n_qubits=n_qubits)
    
    # Build and train the model
    print("Building and training the hybrid quantum-classical model...")
    model = build_hybrid_model(input_shape, quantum_layer)
    
    # Train the model
    history = model.fit(
        X_train_flat, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_flat, y_test),
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_test_flat)
    
    # Inverse transform to get actual prices
    y_test_inv = scaler_y.inverse_transform(y_test)
    predictions_inv = scaler_y.inverse_transform(predictions)
    
    # Calculate metrics
    mse = np.mean((y_test_inv - predictions_inv) ** 2)
    print(f"Mean Squared Error: {mse}")
    
    # Plot results
    plot_predictions(y_test_inv, predictions_inv, title="AAPL Stock Price Prediction")
    
    print("Done!")

if __name__ == "__main__":
    # Fix for TensorFlow and PennyLane integration
    import tensorflow as tf
    main()