from typing import Tuple, List, Any

import numpy as np
import pandas as pd
from keras import Sequential
from matplotlib import pyplot as plt
from numpy import ndarray
from tensorflow import keras
from tensorflow.keras import layers

# Model Constants
FILTER_1 = 32
FILTER_2 = FILTER_1/2
KERNEL_SIZE = 8
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2

# Training Constants
EPOCHS = 20
BATCH_SIZE = 126


def create_autoencoder_model(shape: Tuple) -> keras.Model:
    """
    Create a 1D convolutional autoencoder model.

    Args:
        shape (tuple): Shape of the input data in the form (timestamps, features).

    Returns:
        keras.Model: The compiled autoencoder model.

    Model can be configured by Model Constants:
        FILTER_1 (int): Number of filters for the first convolutional layer.
        FILTER_2 (int): Number of filters for the second convolutional layer.
        KERNEL_SIZE (int): Size of convolutional kernels.
        LEARNING_RATE (float): Learning rate for the Adam optimizer.
        DROPOUT_RATE (float): Dropout rate for Dropout layers.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(shape[1], shape[2])),
            layers.Conv1D(
                filters=FILTER_1, kernel_size=KERNEL_SIZE, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=DROPOUT_RATE),
            layers.Conv1D(
                filters=FILTER_2, kernel_size=KERNEL_SIZE, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=FILTER_2, kernel_size=KERNEL_SIZE, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=DROPOUT_RATE),
            layers.Conv1DTranspose(
                filters=FILTER_1, kernel_size=KERNEL_SIZE, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=KERNEL_SIZE, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def _train(model: Sequential, sequence: np.ndarray) -> keras.callbacks.History:
    """
    Train an autoencoder model for specific sequence.

    Args:
        model (keras.Sequential): The autoencoder model to be trained.
        sequence (Any): Input sequence for training (3D array).
        epochs (int): Number of training epochs.

    Returns:
        keras.callbacks.History: Training history containing loss and metric values.

    Model can be configured by Training Constants:
        BATCH_SIZE (int): Size of batch for training interval
        EPOCHS (int): Number of epochs during training of the model
    """

    history = model.fit(
        sequence,
        sequence,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )

    return history


def train_autoencoder(model: Sequential, sequences: List[np.ndarray]) -> None:
    """
    Train the autoencoder model on multiple sequences and plot overall training history.

    Args:
        model (keras.Sequential): The autoencoder model to be trained.
        sequences (List[np.ndarray]): List of input sequences for training (3D arrays).
        epochs (int): Number of training epochs.
    """
    training_history_val_loss = []
    training_history_loss = []

    for _sequences in sequences:
        history = _train(model, _sequences)
        training_history_val_loss.extend(history.history["val_loss"])
        training_history_loss.extend(history.history["loss"])

    plt.plot(training_history_loss, label="Training Loss")
    plt.plot(training_history_val_loss, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Overall Training and Validation Loss")
    plt.show()


def _calculate_average_loss_threshold(model: Sequential, sequences: List[np.ndarray]) -> float:
    """
    Calculate and plot the histogram of average reconstruction losses across training sequences.

    Args:
        sequences (List[np.ndarray]): List of training sequences.
        model (keras.Model): The autoencoder model for prediction.

    Returns:
        float: Maximum reconstruction error threshold.

    The function calculates the average reconstruction loss for each data point
    across attributes (chunks) in all training sequences. It then plots the histogram
    of average losses and returns the maximum value as the threshold.
    """
    average_train_mae_loss = []

    for x_train in sequences:
        x_train_pred = model.predict(x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=(1, 2))
        average_train_mae_loss.extend(train_mae_loss)

    average_train_mae_loss = np.array(average_train_mae_loss)

    # Plot the histogram of average train MAE loss
    plt.hist(average_train_mae_loss, bins=30)
    plt.xlabel("Average MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    # Plot original and predicted sequences
    x_train = np.mean(sequences[-1][10], axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_train, label="Original Sequence Mean")
    plt.plot(x_train_pred[10], label="Predicted Sequence Mean")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Value")
    plt.title("Comparison of Original and Predicted Sequence Means")
    plt.legend()
    plt.show()

    threshold = np.max(average_train_mae_loss)
    return threshold


def predict_autoencoder(model: Sequential, sequences: List[np.ndarray]) -> tuple[list[Any], ndarray]:
    """
    Predicts sequences using an autoencoder model and calculates the average Mean Absolute Error (MAE) loss for each
    predicted sequence.

    Args:
        model (Sequential): The autoencoder model to be used for sequence prediction.
        sequences (List[np.ndarray]): A list of input sequences (numpy arrays) to be predicted.

    Returns:
        Tuple[List[Any], np.ndarray]: A tuple containing:
            - predicted_sequences (List[Any]): A list of numpy arrays representing the predicted sequences.
            - average_mae_loss (np.ndarray): A 1D numpy array containing the average MAE loss for
             each predicted sequence.

    This function uses an autoencoder model and a list of input sequences to predict each sequence's output.
    It calculates the average Mean Absolute Error (MAE) loss for the predicted sequences.
    The MAE loss is computed for each point in the predicted and original sequences,
     and then averaged across all points in the sequence.

    The function also generates a histogram to visualize the distribution of average MAE losses
     across all predicted sequences.

    Note:
    - The autoencoder model should be compiled and trained before passing it to this function.
    """
    average_mae_loss = []
    predicted_sequences = []

    for sequence in sequences:
        sequence_prediction = model.predict(sequence)
        prediction_mae_loss = np.mean(np.abs(sequence_prediction - sequence), axis=(1, 2))
        average_mae_loss.extend(prediction_mae_loss)
        predicted_sequences.append(sequence_prediction)

    average_mae_loss = np.array(average_mae_loss)
    plot_histogram(average_mae_loss)

    return predicted_sequences, average_mae_loss


def plot_histogram(average_mae_loss):
    plt.hist(average_mae_loss, bins=30)
    plt.xlabel("Average MAE loss")
    plt.ylabel("No of samples")
    plt.show()


def plot_prediction(original_sequence, predicted_sequence):
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(original_sequence, axis=1), label="Original Sequence Mean")
    plt.plot(predicted_sequence, label="Predicted Sequence Mean")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Value")
    plt.title("Comparison of Original and Predicted Sequence Means")
    plt.legend()
    plt.show()


def calculate_threshold(average_mae_loss: np.ndarray) -> float:
    """
    This function takes a 1D numpy array of average MAE loss values and calculates
     the threshold value for anomaly detection.
    The threshold is set as the maximum value among the provided MAE loss values.

    Args:
        average_mae_loss (np.ndarray): A 1D numpy array containing the average MAE loss values.

    Returns:
        float: The calculated threshold value for anomaly detection.
    """
    return np.max(average_mae_loss)


def plot_mae_histogram(individual_mae_loss, attributes):
    """
    Plot a histogram of MAE loss for individual samples.

    Args:
    - individual_mae_loss (list or numpy.ndarray): A list or array containing the MAE loss values for each sample.
    - attributes (str): A label for the legend.

    """
    plt.hist(individual_mae_loss, bins=6, label=attributes)

    plt.xlabel("MAE loss")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.show()


def plot_anomalies(original_df: pd.DataFrame, anomalous_data_indices: List[int]) -> None:
    """
    Plots anomalies in a DataFrame.

    Args:
        original_df (pd.DataFrame): The original DataFrame containing data.
        anomalous_data_indices (List[int]): A list of indices representing anomalous data points.

    Returns:
        None
    """
    subset_df = original_df.iloc[anomalous_data_indices]
    fig, ax = plt.subplots()

    original_df.plot(
        y=['AnalogInVoltage', 'Z-Velocity'],
        ax=ax,
        color=['orange', 'green'],
        label=['Original AnalogInVoltage', 'Original Z-Velocity'],
    )

    subset_df.plot(
        y=['AnalogInVoltage', 'Z-Velocity'],
        ax=ax,
        color=['red', 'red'],
        marker='o',
        label=['Anomalous', ''],
        linestyle='None',
    )

    ax.set_title("Anomalous Data Detection")
    ax.set_xlabel("Index")
    ax.set_ylabel("Values")
    ax.legend()
    plt.show()
