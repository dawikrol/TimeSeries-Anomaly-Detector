import numpy as np

from autoencoder.helpers import create_autoencoder_model, train_autoencoder, \
    predict_autoencoder, calculate_threshold, plot_prediction, plot_mae_histogram, plot_anomalies
from common.helpers import clean_dataframes, read_data_from_directory, normalize_dataframes, create_sequences, SAMPLE_1

# Data Reading
anomaly, normal = read_data_from_directory(SAMPLE_1)

# Data Cleaning
WINDOW_SIZE = 64
KEY_ATTRIBUTES = ["Z-location", "Z-Velocity", "AnalogInVoltage", "AverageFeedRate"]
SMA_ATTRIBUTES = ["Z-Velocity", "AnalogInVoltage"]
cleaned_anomalies = clean_dataframes(anomaly, KEY_ATTRIBUTES, WINDOW_SIZE, SMA_ATTRIBUTES)
cleaned_normals = clean_dataframes(normal, KEY_ATTRIBUTES, WINDOW_SIZE, SMA_ATTRIBUTES)

# Data Preparation
normalized_anomalies, anomalies_means, anomalies_stds = normalize_dataframes(cleaned_anomalies)
normalized_normals, normals_means, normals_stds = normalize_dataframes(cleaned_normals)
sequences_anomalies = create_sequences(normalized_anomalies, WINDOW_SIZE)
sequences_normals = create_sequences(normalized_normals, WINDOW_SIZE)

# Data dividing into training and test sets
training_set = sequences_normals[:-1]
test_normal = sequences_normals[-1]
test_anomalies = sequences_anomalies

# Defining the Autoencoder Model
input_shape = sequences_anomalies[0].shape
autoencoder = create_autoencoder_model(input_shape)
autoencoder.summary()

# Training
train_autoencoder(autoencoder, training_set)
training_predicted_seqs, training_avg_mae_loss = predict_autoencoder(autoencoder, training_set)
test_predicted_seqs, test_avg_mae_loss = predict_autoencoder(autoencoder, test_anomalies)

# Training Visualisation
SAMPLE = -1
SEQ = 10
plot_prediction(training_set[SAMPLE][SEQ], training_predicted_seqs[SAMPLE][SEQ])

# Testing: Setting threshold
threshold = calculate_threshold(training_avg_mae_loss)
test_threshold = calculate_threshold(test_avg_mae_loss)

# Predicting anomalies
for test_predicted, test_original, original_df, raw_df in zip(test_predicted_seqs, test_anomalies, cleaned_anomalies,
                                                              anomaly):
    individual_mae_loss = np.mean(np.abs(test_predicted - test_original), axis=1)
    plot_mae_histogram(individual_mae_loss, KEY_ATTRIBUTES)
    anomalies = individual_mae_loss > threshold

    anomalous_data_indices = []
    df_test_value = original_df.values
    for data_idx in range(WINDOW_SIZE - 1, len(df_test_value) - WINDOW_SIZE + 1):
        if np.any(anomalies[data_idx - WINDOW_SIZE + 1: data_idx]):
            anomalous_data_indices.append(data_idx)

    plot_anomalies(original_df, anomalous_data_indices)
