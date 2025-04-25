import time
import csv
import threading
import os
import joblib
import numpy as np
from collections import defaultdict, deque
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

import socket  # For UDP communication with Unity

# UDP configuration for Unity
UNITY_IP = "127.0.0.1"  # Localhost - change if Unity runs on another machine
UNITY_PORT = 8052       # Must match Unity's listening port
unity_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Cognitive load tracking
last_predictions = deque(maxlen=10)  # Stores last 10 predictions
ALERT_THRESHOLD = 9                 # Number of consecutive matches needed

# Configuration
MODEL_DIR = "models"
SCALER_DIR = "scalers"
ENCODER_DIR = "encoders"

# Buffer for storing incoming data
data_buffer = defaultdict(lambda: deque(maxlen=500))

# Lock for thread-safe access
lock = threading.Lock()

# Output CSV files
csv_file = "emotibit_data.csv"
cogload_file = "cognitive_load_data.csv"
prediction_file = "cognitive_load_predictions.csv"

# Field names
fieldnames = [
    "timestamp", "ACC:X", "ACC:Y", "ACC:Z",
    "PPG:RED", "PPG:IR", "PPG:GRN", "EDA", "HUMIDITY", "TEMP", "T1", "HR", "BVA",
    "GYRO:X", "GYRO:Y", "GYRO:Z", "MAG:X", "MAG:Y", "MAG:Z"
]

# Load ML components
def load_ml_components():
    """Load the trained model, scalers, and label encoder"""
    try:
        clf = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
        std_scaler = joblib.load(os.path.join(MODEL_DIR, 'standard_scaler.pkl'))
        bva_scaler = joblib.load(os.path.join(SCALER_DIR, 'bva_scaler.pkl'))
        label_encoder = joblib.load(os.path.join(ENCODER_DIR, 'label_encoder.pkl'))
        return clf, std_scaler, bva_scaler, label_encoder
    except Exception as e:
        print(f"Error loading ML components: {e}")
        return None, None, None, None

clf, std_scaler, bva_scaler, label_encoder = load_ml_components()

# OSC Handler
def generic_handler(address, *args):
    signal = address.split("/")[-1]  # e.g., ACC:X
    value = args[0]
    if signal == "TEMP2":
        signal = "T1"
    with lock:
        data_buffer[signal].append(value)

def compute_bva(ppg_values):
    """Compute Blood Volume Amplitude from PPG signal"""
    if len(ppg_values) < 10:
        return None

    ppg_array = np.array(ppg_values)
    peaks, _ = find_peaks(ppg_array)
    troughs, _ = find_peaks(-ppg_array)

    if len(peaks) == 0 or len(troughs) == 0:
        return None

    last_peak = ppg_array[peaks[-1]]
    last_trough = ppg_array[troughs[-1]]
    return abs(last_peak - last_trough)

def predict_cognitive_load(eda, temp, bva):
    """Predict with proper feature name handling"""
    if None in (clf, std_scaler, bva_scaler, label_encoder):
        return "MODEL_NOT_LOADED"
    
    try:
        # Create DataFrame to preserve feature names
        import pandas as pd
        input_data = pd.DataFrame([[eda, temp, bva]], 
                                columns=['EDA', 'Temp', 'BVA'])
        
        # 1. Scale BVA between 0 and 1
        input_data['BVA'] = bva_scaler.transform(input_data[['BVA']])
        
        # 2. Scale all features
        features_scaled = std_scaler.transform(input_data)
        
        # 3. Predict and decode label
        prediction_num = clf.predict(features_scaled)[0]
        return label_encoder.inverse_transform([prediction_num])[0]
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return "PREDICTION_ERROR"

def initialize_files():
    """Initialize output files with headers"""
    # Main data file
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # Cognitive load data file
    with open(cogload_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EDA", "Temp", "BVA"])
    
    # Prediction file
    with open(prediction_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "EDA", "Temp", "BVA", "CognitiveLoad"])

def aggregate_and_save():
    """Process data and save results every 2 seconds"""
    initialize_files()
    
    # Initialize MinMaxScaler for BVA (0-1)
    bva_scaler = MinMaxScaler(feature_range=(0, 1))
    
    while True:
        
        time.sleep(2)
        timestamp = time.time()
        row = {"timestamp": timestamp}
        prediction_data = {}

        with lock:
            # Compute BVA first using PPG:IR
            ppg_ir_values = list(data_buffer["PPG:IR"])
            bva_value = compute_bva(ppg_ir_values)
            
            # Scale BVA between 0 and 1
            if bva_value is not None:
                bva_scaled = bva_scaler.fit_transform([[bva_value]])[0][0]
            else:
                bva_scaled = None
                
            row["BVA"] = bva_value  # Original value in main data file
            prediction_data["BVA"] = bva_scaled  # Scaled value for cognitive load prediction

            # Get other values
            for key in fieldnames[1:]:
                if key == "BVA":
                    continue
                if data_buffer[key]:
                    row[key] = sum(data_buffer[key]) / len(data_buffer[key])
                    data_buffer[key].clear()
                else:
                    row[key] = None

            # Clear PPG:IR after computing BVA
            data_buffer["PPG:IR"].clear()

            # Prepare cognitive load data
            prediction_data["EDA"] = row["EDA"]
            prediction_data["Temp"] = row["T1"]  # Using T1 as temperature

        # Save to main data file (with original BVA)
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        # Save cognitive load features if all data available
        if None not in (prediction_data["EDA"], prediction_data["Temp"], prediction_data["BVA"]):
            # Save features with scaled BVA
            with open(cogload_file, mode="a", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow([prediction_data["EDA"], prediction_data["Temp"], prediction_data["BVA"]])
            
            # Make and save prediction
            cognitive_load = predict_cognitive_load(
                prediction_data["EDA"],
                prediction_data["Temp"],
                prediction_data["BVA"]  # Already scaled
            )
            
            # Save prediction with scaled BVA
            with open(prediction_file, mode="a", newline="") as pf:
                writer = csv.writer(pf)
                writer.writerow([timestamp, prediction_data["EDA"], 
                               prediction_data["Temp"], prediction_data["BVA"], 
                               cognitive_load])
            
            print(f"Predicted cognitive load: {cognitive_load}")

            # Update prediction history
            last_predictions.append(cognitive_load)
            
            # Check for 10 consecutive similar predictions
            if len(last_predictions) >= 10:
                unique_preds = set(last_predictions)
                if len(unique_preds) == 1:  # All predictions are the same
                    alert_message = f"ALERT|{cognitive_load.upper()}"
                    send_unity_alert(alert_message)
                    last_predictions.clear()  # Reset after alert

def send_unity_alert(message):
    """Send alert message to Unity via UDP"""
    try:
        unity_socket.sendto(message.encode(), (UNITY_IP, UNITY_PORT))
        print(f"Sent Unity alert: {message}")
    except Exception as e:
        print(f"Failed to send Unity alert: {e}")


# Start aggregator thread
threading.Thread(target=aggregate_and_save, daemon=True).start()

# Set up OSC server
dispatcher = Dispatcher()
dispatcher.set_default_handler(generic_handler)

ip = "127.0.0.1"
port = 12346  # Match with EmotiBit XML
server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)

print(f"Listening on {ip}:{port}")
print("Data collection and cognitive load prediction running...")
server.serve_forever()