import random

from keras.src.optimizers.adam import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from lstm.helpers import extract_features_and_target_values, create_sequences, evaluate_and_visualize_results
from common.helpers import clean_dataframes, read_data_from_directory, SAMPLE_1, FEATURES_ATTRIBUTES, KEY_ATTRIBUTES, \
    SMA_WINDOW_SIZE, SMA_ATTRIBUTES

# Data Reading
anomalies, normal = read_data_from_directory(SAMPLE_1)

# Data Cleaning
cleaned_anomalies = clean_dataframes(anomalies, KEY_ATTRIBUTES, SMA_WINDOW_SIZE, SMA_ATTRIBUTES)
cleaned_normals = clean_dataframes(normal, KEY_ATTRIBUTES, SMA_WINDOW_SIZE, SMA_ATTRIBUTES)

# Data Dividing into training and test datasets
validation_data_set = [cleaned_anomalies.pop()] + [cleaned_normals.pop()]
training_anomalies, test_anomalies = train_test_split(cleaned_anomalies, test_size=0.05)
training_normals, test_normals = train_test_split(cleaned_normals, test_size=0.05)
training_data_set = training_anomalies + training_normals
test_data_set = test_anomalies + test_normals
random.shuffle(training_data_set)

# Extracting features and target variables from dataframes. Normalizing features.
train_features, train_target = extract_features_and_target_values(training_data_set)
test_features, test_target = extract_features_and_target_values(test_data_set)
predict_features, predict_target = extract_features_and_target_values(validation_data_set)

# Data Structuring
WINDOWS_SIZE = 120

train_sequences = []
train_labels = []
for feature, target in zip(train_features, train_target):
    sequences_batch, labels_batch = create_sequences(feature, target, WINDOWS_SIZE)
    train_sequences.append(sequences_batch)
    train_labels.append(labels_batch)

test_sequences = []
test_labels = []
for feature, target in zip(test_features, test_target):
    sequences_batch, labels_batch = create_sequences(feature, target, WINDOWS_SIZE)
    test_sequences.append(sequences_batch)
    test_labels.append(labels_batch)

predict_sequences = []
predict_labels = []
for feature, target in zip(predict_features, predict_target):
    sequences_batch, labels_batch = create_sequences(feature, target, WINDOWS_SIZE)
    predict_sequences.append(sequences_batch)
    predict_labels.append(labels_batch)

# Model Architecture
BATCH_SIZE = 32
NEURONS = 64
LEARNING_RATE = 0.0001

model = Sequential()
model.add(LSTM(NEURONS, input_shape=(WINDOWS_SIZE, len(FEATURES_ATTRIBUTES))))
model.add(Dense(1, activation='sigmoid'))

# Model Training
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
for sequences, label in zip(train_sequences, train_labels):
    model.fit(sequences, label, epochs=10, batch_size=BATCH_SIZE)

# Model Evaluation
for sequences, label in zip(test_sequences, test_labels):
    loss, accuracy = model.evaluate(sequences, label)
    print(f"Test loss: {loss:.6f}")
    print(f"Test accuracy: {accuracy:.4f}")

# Model Prediction
predictions = []
binary_predictions = []
for sequences, label in zip(predict_sequences, predict_labels):
    prediction = model.predict(sequences)
    predictions.append(prediction)

    binary_prediction = prediction.round()
    binary_predictions.append(binary_prediction)

# Show Results
for b_prediction, origin_df in zip(binary_predictions, validation_data_set):
    evaluate_and_visualize_results(origin_df, b_prediction, WINDOWS_SIZE)
