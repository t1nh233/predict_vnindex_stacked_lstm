import pandas as pd
import numpy as np
import torch
import json
import os
import sys
import joblib
import argparse

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from model import LSTMModel
    from src.utils import load_and_process_data, feature_extraction
except ImportError:
    sys.exit(1)

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_vnindex_lstm.pth')
CONFIG_PATH = os.path.join(MODEL_DIR, 'best_params.json')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl') 

DEVICE = torch.device('cpu')

def main(input_file):
    if not os.path.exists(input_file):
        print("Khong tim thay file input")
        return
    
    for f in [MODEL_PATH, CONFIG_PATH, SCALER_PATH]:
        if not os.path.exists(f):
            print("Khong tim thay cac file train")
            return

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            feature_cols = config.get('feature_columns', [])
            window_size = config.get('window_size', 30)

            scaler = joblib.load(SCALER_PATH)

            model = LSTMModel(
                input_size=len(feature_cols),
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout_rate=config['dropout_rate']
            ).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
        
    except Exception:
        print("Lỗi khi tải model hoặc config")
        return

    try:
        df = load_and_process_data(input_file)
        df = feature_extraction(df, create_labels=False)

        if len(df) < window_size:
            return

        last_window = df.tail(window_size).copy()
        last_date = last_window.index[-1].strftime('%d/%m/%Y')

        n_features_expected = scaler.n_features_in_
        needed_cols = feature_cols + ['Label']
        if 'Label' not in last_window.columns:
            last_window['Label'] = 0.0

        input_raw = last_window[needed_cols].values
        if input_raw.shape[1] != n_features_expected:
            print("Dữ liệu đầu vào không đúng định dạng")
            return
        
        input_scaled = scaler.transform(input_raw)
        X_input = input_scaled[:, :-1] 
        X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    except Exception:
        print("Lỗi khi xử lý dữ liệu đầu vào")
        return

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()[0][0]

    if 'Close' in feature_cols:
        close_idx = feature_cols.index('Close')
        min_val = scaler.data_min_[close_idx]
        max_val = scaler.data_max_[close_idx]
        final_price = y_pred_scaled * (max_val - min_val) + min_val
        last_close = last_window['Close'].iloc[-1]
        change = final_price - last_close
        return {
            "date": last_date,
            "last_close": last_close,
            "predicted_close": final_price,
            "change": change
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dự báo VN-Index')
    parser.add_argument('--input', type=str, required=True, help='Đường dẫn file CSV input')
    args = parser.parse_args()
    result = main(args.input)
    if result:
        print(result)
