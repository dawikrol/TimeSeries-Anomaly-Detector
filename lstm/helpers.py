import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.helpers import TARGET_COLUMN, FEATURES_ATTRIBUTES
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def line_plot(df, column_name: str):
    plt.plot(df[column_name])
    plt.xlabel('Time Index')
    plt.ylabel(column_name)
    if not df.index[df["BT-Detect"] == 1].empty:
        bt_detected_start_index = df.index[df["BT-Detect"] == 1][0]
        print(f"INDEX: {bt_detected_start_index}")
        plt.axvline(x=bt_detected_start_index, color='g', linestyle='--', label='BT-Detect')

    plt.legend()
    plt.show()


def extract_features_and_target_values(dataframes: list[pd.DataFrame]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extracts the features and target values from a list of pandas DataFrames. Normalizes features.
    KEY_ATTRIBUTES = ["Z-location", "Z-Velocity", "AnalogInVoltage", "BT-VelRatio", "BT-VoltageRatio",
                        "BT-VoltageStdDev", "AverageFeedRate"]
    TARGET_ATTRIBUTES = ["BT-Detect"]

    Args:
        dataframes: A list of pandas DataFrames containing the data.

    Returns:
        A tuple containing the features and target values as separate lists.
    """
    features = []
    targets = []

    for df in dataframes:
        try:
            df_features = df[FEATURES_ATTRIBUTES].values
            df_target = df[TARGET_COLUMN].values

            scaler = StandardScaler()
            df_features = scaler.fit_transform(df_features)

            features.append(df_features)
            targets.append(df_target)
        except KeyError as e:
            print(f"Error: DataFrame is missing required columns. {e}")

    return features, targets


def create_sequences(features: np.ndarray, target: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences and labels from the given features and target arrays.

    Args:
        features: The input features array of shape (num_samples, num_features).
        target: The target array of shape (num_samples,).
        window_size: The size of the input window for creating sequences.

    """
    sequences = []
    labels = []

    if len(features) != len(target):
        raise ValueError("Features and target arrays must have the same length.")

    if window_size >= len(features):
        raise ValueError("Window size must be smaller than the length of features array.")

    for i in range(len(features) - window_size):
        sequence = features[i:i + window_size]
        label = target[i + window_size]
        sequences.append(sequence)
        labels.append(label)

    return np.array(sequences), np.array(labels)


def evaluate_and_visualize_results(origin_df: pd.DataFrame, anomalies_predictions: np.ndarray,
                                   windows_size: int) -> None:
    """
    Evaluate and visualize the results of anomalies predictions.

    Args:
        origin_df (pd.DataFrame): The original DataFrame containing data.
        anomalies_predictions (np.ndarray): Binary predictions.
        windows_size (int): Size of the window for predictions.

    Returns:
        None
    """
    # Transform results to original format
    corrected_prediction = [0] * windows_size + [int(x) for x in anomalies_predictions.flatten().tolist()]

    # Data visualization
    origin_df["BT-Predicted"] = corrected_prediction
    line_plot(origin_df, 'BT-Predicted')

    # Compute evaluation metrics
    cm = confusion_matrix(origin_df['BT-Detect'], corrected_prediction)
    accuracy = accuracy_score(origin_df['BT-Detect'], corrected_prediction)
    precision = precision_score(origin_df['BT-Detect'], corrected_prediction)
    recall = recall_score(origin_df['BT-Detect'], corrected_prediction)
    f1 = f1_score(origin_df['BT-Detect'], corrected_prediction)

    # Print evaluation metrics
    print("Confusion Matrix")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
