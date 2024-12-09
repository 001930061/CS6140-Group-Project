# README: Stock Prediction Using Recurrent Neural Networks

## Overview
This project implements multiple neural network models to predict stock prices based on time-series data. The models are built using TensorFlow/Keras and leverage different types of recurrent neural networks (RNNs), including LSTM, GRU, and general RNN architectures. Each architecture is tested in two configurations: forward-only and bidirectional (forward-backward). The goal is to compare the predictive accuracy of these models under consistent settings.

---

## Models Implemented
1. **LSTM_Forward**: Long Short-Term Memory network with a forward-only configuration.
2. **LSTM_Forward_Backward**: Bidirectional LSTM to capture dependencies in both forward and backward directions.
3. **GRU_Forward**: Gated Recurrent Unit network in a forward-only configuration.
4. **GRU_Forward_Backward**: Bidirectional GRU for capturing bidirectional dependencies.
5. **General_RNN_Forward**: Standard RNN model with a forward-only setup.
6. **General_RNN_Forward_Backward**: Bidirectional RNN for processing sequential data in both directions.

---

## Neural Network Design
Each model consists of the following components:
- **LSTM/GRU/RNN Layers**: Configurable number of recurrent layers with specified units.
- **Dropout Layer**: Prevents overfitting by randomly dropping connections during training.
- **Dense Output Layer**: Produces predictions for the next time step.
- **Optimizer**: Adam optimizer is used for efficient gradient descent.
- **Loss Function**: Mean squared error (MSE) measures prediction accuracy.

The bidirectional models utilize bidirectional wrappers to process input sequences from both forward and backward directions.

---

## Training and Prediction Workflow

### 1. **Model Creation**
The `Model` class initializes an RNN-based network with the given hyperparameters:
- **Number of layers** (`num_layers`)
- **Size of each layer** (`size_layer`)
- **Learning rate** (`learning_rate`)
- **Dropout rate** (`forget_bias`)
- **Input and output size**

### 2. **Training**
The `train_model` function:
- Trains the model using a time-series dataset.
- Divides the data into overlapping sequences of a specified length (`timestamp`).
- Iterates through multiple epochs, calculating the loss and accuracy for each batch.
- Uses backpropagation to update weights.

### 3. **Prediction**
The `predict` function:
- Generates predictions for both the training data and future time steps.
- Smooths the output using a weighted anchor function to reduce noise.
- Reverses scaling (if applied) for interpretability of results.

### 4. **Forecasting**
The `forecast` function orchestrates:
- Model initialization.
- Training on the provided dataset.
- Prediction of future stock prices.

---

## Key Functions

### `create_model`
Initializes the model with the specified hyperparameters:
- Learning rate, number of layers, size of each layer, dropout rate, and input-output dimensions.

### `train_model`
Processes training data in batches:
- Extracts input-output sequences of length `timestamp`.
- Updates weights using the Adam optimizer and tracks loss and accuracy.

### `predict`
Uses the trained model to:
- Predict stock prices for training data.
- Extrapolate future stock prices based on the last known values.

### `calculate_accuracy`
Computes prediction accuracy as a percentage:
- Compares real vs. predicted values using a normalized percentage error formula.

### `anchor`
Smoothens predictions using a weighted average:
- Reduces abrupt changes in predictions for better interpretability.

### `forecast`
Executes the entire pipeline:
- Creates, trains, and uses the model for future predictions.

---

## Data Requirements
- Input data should be a time-series dataset.
- Ensure proper normalization/scaling before feeding data into the model.
- The first column should represent dates for tracking predictions.

---

## Usage
1. **Prepare your dataset**:
   - Ensure it is formatted correctly (e.g., dates in the first column, features in subsequent columns).
   - Normalize or scale the data for compatibility with the model.

2. **Set Hyperparameters**:
   - Configure `learning_rate`, `num_layers`, `size_layer`, `dropout_rate`, `epoch`, and `timestamp` values.

3. **Train and Predict**:
   - Call `forecast()` to train the model and predict future stock prices.

4. **Evaluate Results**:
   - Compare the accuracy of each model (forward-only vs. bidirectional).

---

## Comparison of Models
- **Forward-Only Models**: Focus on learning patterns in one direction, suitable for short-term predictions.
- **Bidirectional Models**: Leverage dependencies in both past and future directions, often yielding better accuracy for complex sequences.

---

## Dependencies
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib (for visualization)
- tqdm (for progress bars)

---

## Future Work
- Experiment with additional hyperparameter tuning.
- Test models on diverse datasets.
- Incorporate attention mechanisms for better sequence learning.

---

## License
None
