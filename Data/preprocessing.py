import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import joblib
import os

# ==== File Paths ====
INPUT_EXCEL_FILE = 'dft_traffic_counts_raw_counts.csv'  
PREPROCESSED_CSV_FILE = 'preprocessed_traffic.csv'
ENCODER_FILE = 'onehot_encoder.joblib'
SCALER_FILE = 'standard_scaler.joblib'
TARGET_SCALER_FILE = 'target_scaler.joblib'

# ==== Constants ====
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# ==== Load Dataset ====
try:
    df = pd.read_csv(INPUT_EXCEL_FILE, low_memory=False)
    print(f"Loaded data from {INPUT_EXCEL_FILE}")
    print("Initial Shape:", df.shape)
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(f"Error loading Excel: {e}")
    exit()

# ==== Datetime Conversion ====
try:
    df['count_date'] = pd.to_datetime(df['count_date'], errors='coerce')
    df.dropna(subset=['count_date'], inplace=True)
    df['Year'] = df['count_date'].dt.year
    df['Month'] = df['count_date'].dt.month
    df['DayOfWeek'] = df['count_date'].dt.dayofweek
    df['DayOfYear'] = df['count_date'].dt.dayofyear
except Exception as e:
    print(f"Datetime conversion failed: {e}")
    exit()

# ==== Target Variable ====
target_variable = 'cars_and_taxis'

# ==== Categorical and Numerical Features ====
categorical_features = ['direction_of_travel', 'region_name', 'local_authority_name', 'road_type']
numerical_features = ['hour', 'link_length_km', 'latitude', 'longitude', 'Month', 'DayOfWeek', 'DayOfYear']
required_columns = categorical_features + numerical_features + [target_variable]

# ==== Check Missing Columns ====
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print("Missing required columns:", missing)
    exit()

# ==== Filter Required Columns ====
df_processed = df[required_columns].copy()

# ==== One-Hot Encode Categorical Features ====
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df_processed[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features), index=df_processed.index)

# Save Encoder
joblib.dump(encoder, ENCODER_FILE)
print(f"Saved OneHotEncoder to {ENCODER_FILE}")

# ==== Scale Numerical Features ====
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df_processed[numerical_features])
scaled_num_df = pd.DataFrame(scaled_nums, columns=numerical_features, index=df_processed.index)

# Save Scaler
joblib.dump(scaler, SCALER_FILE)
print(f"Saved StandardScaler to {SCALER_FILE}")

# ==== Scale Target (Optional but useful) ====
target_scaler = MinMaxScaler()
scaled_target = target_scaler.fit_transform(df_processed[[target_variable]])
scaled_target_df = pd.DataFrame(scaled_target, columns=[target_variable + '_scaled'], index=df_processed.index)

# Save Target Scaler
joblib.dump(target_scaler, TARGET_SCALER_FILE)
print(f"Saved Target MinMaxScaler to {TARGET_SCALER_FILE}")

# ==== Final DataFrame ====
df = df.sample(frac=0.1, random_state=42)  # Use only 10% of data


# ==== Save Preprocessed Data ====
df.to_csv(PREPROCESSED_CSV_FILE, index=False)
print(f"Saved preprocessed data to {PREPROCESSED_CSV_FILE}")
print("Final shape:", df.shape)
