"""
Main Application

This script demonstrates the integration of quantum computing with machine learning
for financial forecasting using the implemented modules.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from financial_data_processor import FinancialDataProcessor
from hybrid_quantum_model import HybridQuantumModel
import tensorflow as tf

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configure output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = output_dir

def main():
    """Main function to run the quantum ML financial forecasting demo."""
    print("Quantum ML Financial Forecasting Demo")
    print("=====================================")
    
    # Initialize the data processor
    print("\nInitializing data processor...")
    data_processor = FinancialDataProcessor()
    
    # Fetch stock data
    ticker = "AAPL"
    print(f"\nFetching historical data for {ticker}...")
    data = data_processor.fetch_stock_data(ticker=ticker, period="2y")
    print(f"Fetched data shape: {data.shape}")
    
    # Add technical indicators
    print("\nAdding technical indicators...")
    data_with_indicators = data_processor.add_technical_indicators()
    print(f"Features available: {data_with_indicators.columns.tolist()}")
    
    # Plot the stock data with indicators
    print("\nGenerating stock analysis plot...")
    try:
        data_processor.plot_stock_data()
        print("Stock analysis plot generated successfully.")
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")
    
    # Prepare sequences for ML
    print("\nPreparing data sequences for training...")
    sequence_length = 10
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'MA5', 'MA20', 'RSI', 'MACD', 'Signal']
    X_train, X_test, y_train, y_test = data_processor.prepare_sequences(
        data_with_indicators, 
        feature_columns=feature_columns,
        target_column='Close',
        sequence_length=sequence_length
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Initialize the hybrid quantum-classical model
    print("\nInitializing hybrid quantum-classical model...")
    n_qubits = 4
    hybrid_model = HybridQuantumModel(n_qubits=n_qubits, n_layers=2)
    
    # Build the model
    print("\nBuilding model architecture...")
    try:
        # Use a small subset for testing in Docker
        X_train_small = X_train[:10]
        y_train_small = y_train[:10]
        X_test_small = X_test[:5]
        y_test_small = y_test[:5]
        
        model = hybrid_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.summary()
        
        # Train the model
        print("\nTraining the hybrid quantum-classical model...")
        print("Using small subset for demonstration purposes...")
        history = hybrid_model.fit(
            X_train_small, y_train_small,
            epochs=2,  # Reduced for faster execution
            batch_size=5,
            validation_data=(X_test_small, y_test_small),
            verbose=1
        )
        
        # Plot training history
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'training_history.png'))
            plt.close()
            print(f"\nTraining history plot saved to {output_dir}/training_history.png")
        except Exception as e:
            print(f"Warning: Could not generate training history plot: {e}")
        
        # Make predictions
        print("\nGenerating predictions...")
        predictions = hybrid_model.predict(X_test_small)
        
        # Inverse transform to get actual prices
        y_test_inv = data_processor.inverse_transform_y(y_test_small)
        predictions_inv = data_processor.inverse_transform_y(predictions)
        
        # Calculate metrics
        mse = np.mean((y_test_inv - predictions_inv) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_inv - predictions_inv))
        
        print(f"\nPerformance Metrics:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # Plot results
        print("\nGenerating prediction plot...")
        try:
            data_processor.plot_predictions(
                y_test_inv, 
                predictions_inv, 
                title=f"{ticker} Stock Price Prediction"
            )
            print(f"Prediction plot saved to {output_dir}/prediction_results.png")
        except Exception as e:
            print(f"Warning: Could not generate prediction plot: {e}")
        
        print("\nDemo completed successfully!")
    
    except Exception as e:
        print(f"\nError during model execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()