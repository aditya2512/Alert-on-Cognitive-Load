

```markdown
# EmotiBit Cognitive Load Prediction System

## Overview
This system collects physiological data from an EmotiBit device via OSC, processes it in real-time, and predicts cognitive load levels using machine learning. It includes alert functionality for sustained cognitive load states.

## Features
- Real-time OSC data collection from EmotiBit
- Physiological signal processing (PPG, EDA, temperature)
- Blood Volume Amplitude (BVA) calculation
- Cognitive load prediction using Random Forest
- Alert system for consistent patterns
- CSV data logging
- Unity integration via UDP

## Requirements
- Python 3.7+
- Required packages:
  ```bash
  pip install python-osc scikit-learn scipy numpy joblib
  ```

## Setup
1. Create these directories:
   ```
   models/
   scalers/ 
   encoders/
   ```
2. Place these files in respective directories:
   - `random_forest_model.pkl` in `models/`
   - `bva_scaler.pkl` in `scalers/`
   - `label_encoder.pkl` in `encoders/`

## Usage
```bash
python emotibit_cognitive_load.py
```

## Configuration
| Parameter          | Default Value    | Description                          |
|--------------------|------------------|--------------------------------------|
| `UNITY_IP`         | 127.0.0.1       | Unity IP address                    |
| `UNITY_PORT`       | 8052            | Unity UDP port                      |
| `ALERT_THRESHOLD`  | 10              | Consecutive predictions for alert   |
| OSC Server Port    | 12346           | EmotiBit OSC input port             |

## Output Files
| File                          | Contents                                |
|-------------------------------|-----------------------------------------|
| `emotibit_data.csv`           | Raw physiological data                  |
| `cognitive_load_data.csv`     | Processed features for prediction       |
| `cognitive_load_predictions.csv` | Timestamped prediction results        |

## Unity Integration
The system sends UDP alerts in format:
```
ALERT|COGNITIVE_LOAD_LEVEL
```

## Troubleshooting
- **Model loading errors**: Verify all .pkl files are in correct directories
- **OSC connection issues**: Check EmotiBit is sending to correct IP:port
- **Prediction errors**: Ensure all three features (EDA, Temp, BVA) are available

## License
MIT License
```
