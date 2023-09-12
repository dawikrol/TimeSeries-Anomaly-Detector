import glob
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd


SOURCE_DIR = pathlib.Path("")
SAMPLE_1 = SOURCE_DIR.joinpath("SAMPLE_1")

SMA_ATTRIBUTES = ["Z-Velocity", "AnalogInVoltage", "BT-VoltageStdDev"]
SMA_WINDOW_SIZE = 50
INPUT_COLUMNS = ["AnalogInVoltage", "Z-Velocity", "BT-VoltageStdDev"]
TARGET_COLUMN = ["BT-Detect"]

KEY_ATTRIBUTES = ["Z-location",
                  "Z-Velocity",
                  "AnalogInVoltage",
                  "BT-Detect",
                  "BT-VelRatio",
                  "BT-VoltageRatio",
                  "BT-VoltageStdDev",
                  "AverageFeedRate"
                  ]

FEATURES_ATTRIBUTES = [attr for attr in KEY_ATTRIBUTES if attr not in TARGET_COLUMN]


def read_data_from_directory(directory: pathlib.Path) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Read data from CSV files in the DETECTED and NOT_DETECTED subdirectories of the given directory.

    Args:
        directory (Path): Path to the main directory containing DETECTED and NOT_DETECTED subdirectories.

    Returns:
        tuple: A tuple containing two lists: detected and not_detected.
               The detected list contains dataframes read from files in DETECTED subdirectory.
               The not_detected list contains dataframes read from files in NOT_DETECTED subdirectory.
    """
    detected: list[pd.DataFrame] = []
    not_detected: list[pd.DataFrame] = []

    detected_src = directory / "DETECTED"
    not_detected_src = directory / "NOT_DETECTED"

    try:
        for file in glob.glob(f"{detected_src}/*"):
            df = pd.read_csv(file)
            detected.append(df)
    except Exception as e:
        print(f"Error occurred while reading files in DETECTED subdirectory: {e}")

    try:
        for file in glob.glob(f"{not_detected_src}/*"):
            df = pd.read_csv(file)
            not_detected.append(df)
    except Exception as e:
        print(f"Error occurred while reading files in NOT_DETECTED subdirectory: {e}")

    return detected, not_detected


def clean_dataframes(dataframes: List[pd.DataFrame], key_attributes: List[str], sma_window_size: int,
                     sma_attributes: List[str]) -> list[pd.DataFrame]:
    """
    Cleans a list of DataFrames by performing various data preprocessing steps:
        1. Drop NaN values
        2. Select only specified KEY_ATTRIBUTES within each dataframe and drop rows with missing values.
        3. Apply Simple Moving Average (SMA) filter for specified SMA_ATTRIBUTES.
        4. Trim leading and trailing rows after preprocessing.

    Args:
        dataframes (list[pd.DataFrame]): List of input DataFrames to be cleaned.
        key_attributes (list[str]): List of column names representing the key attributes to be selected.
        sma_window_size (int): Window size for the Simple Moving Average (SMA) filter.
        sma_attributes (list[str]): List of column names to which the SMA filter will be applied.

    Returns:
        list[pd.DataFrame]: List of cleaned DataFrames after preprocessing.

    Raises:
        ValueError: If `dataframes` is not a list or contains non-pandas DataFrame objects.
        Exception: Any other exceptions raised during the cleaning process will be caught and printed.
    """
    cleaned_dataframes = []
    for df in dataframes:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input dataframes must be of type pd.DataFrame")

        try:
            cleaned_df = df[key_attributes].dropna()
            cleaned_sma_df = sma_filter(cleaned_df, sma_window_size, sma_attributes)
            trimmed_df = trim_dataframe(cleaned_sma_df)
            cleaned_dataframes.append(trimmed_df)
        except Exception as e:
            print(f"Error occurred while cleaning DataFrame: {e}")

    return cleaned_dataframes


def sma_filter(df: pd.DataFrame, window_size: int, columns: List[str]) -> pd.DataFrame:
    """
    Apply Simple Moving Average (SMA) filter to the specified columns of a DataFrame.

    Args:
        df: Input DataFrame.
        window_size: Size of the moving window used for the SMA calculation.
        columns: List of column names to apply the SMA filter on.

    Returns:
        A new DataFrame with the specified columns replaced by their SMA values.

    """
    df_copy = df.copy()
    for column_name in columns:
        windows = df[column_name].rolling(window_size)
        df_copy[column_name] = windows.mean()[window_size - 1:]
    return df_copy


def trim_dataframe(df: pd.DataFrame, trim_head_ratio: float = 0.25, trim_tail_ratio: float = 0.1) -> pd.DataFrame:
    """
    Trims the specified ratios of rows from the head and tail of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be trimmed.
        trim_head_ratio (float, optional): The ratio of rows to be trimmed from the head of the df. Defaults to 0.25.
        trim_tail_ratio (float, optional): The ratio of rows to be trimmed from the tail of the df. Defaults to 0.1.

    Returns:
        pd.DataFrame: The trimmed DataFrame.

    Raises:
        ValueError: If `df` is not a pandas DataFrame.
        ValueError: If `trim_head_ratio` or `trim_tail_ratio` are not within the range [0, 1].

    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")

    if not (0 <= trim_head_ratio <= 1) or not (0 <= trim_tail_ratio <= 1):
        raise ValueError("trim_head_ratio and trim_tail_ratio must be within the range [0, 1].")

    num_rows_to_cut_head = int(trim_head_ratio * len(df))
    num_rows_to_cut_tail = int(trim_tail_ratio * len(df))

    try:
        trimmed_df = df.iloc[num_rows_to_cut_head:-num_rows_to_cut_tail].copy()
    except Exception as e:
        raise ValueError("Error occurred while trimming DataFrame: " + str(e))

    return trimmed_df


def normalize_dataframes(dataframes: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[pd.Series], List[pd.Series]]:
    """
    Normalizes a list of DataFrames by calculating training means and standard deviations,
    and then applying normalization to each DataFrame.

    Args:
        dataframes (List[pd.DataFrame]): List of input DataFrames to be normalized.

    Returns:
        Tuple[List[pd.DataFrame], List[pd.Series], List[pd.Series]]:
            - List of normalized DataFrames.
            - List of Series containing training means for each DataFrame.
            - List of Series containing training standard deviations for each DataFrame.
    """
    training_means = []
    training_stds = []
    normalized_dataframes = []

    for df in dataframes:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input dataframes must be of type pd.DataFrame")

        # Calculate training mean and std
        training_mean = df.mean()
        training_std = df.std()
        training_means.append(training_mean)
        training_stds.append(training_std)

        # Normalize the current DataFrame
        normalized_df = (df - training_mean) / training_std
        normalized_dataframes.append(normalized_df)

    return normalized_dataframes, training_means, training_stds


def create_sequences(dataframes: List[pd.DataFrame], time_steps) -> List[np.ndarray]:
    """
    Creates sequences from a list of DataFrames for Autoresponder model.

    Args:
        dataframes (List[pd.DataFrame]): A list of pandas DataFrames containing the data.
        time_steps (int, optional): The number of time steps in each sequence. Defaults to WINDOWS_SIZE.

    Returns:
        List[np.ndarray]: A list of numpy arrays containing sequences of data.
    """
    sequences = []

    for df in dataframes:
        output = []
        df_values = df.values
        num_rows = len(df_values)

        for i in range(num_rows - time_steps + 1):
            sequence = df_values[i: (i + time_steps)]
            output.append(sequence)
        sequences.append(np.stack(output))

    return sequences
