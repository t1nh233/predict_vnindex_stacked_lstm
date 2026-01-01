import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def convert_volume(vol_str):
    multiplier_dict = {"K": 1000, "M": 1000000, "B": 1000000000}

    if not isinstance(vol_str, str):
        return vol_str

    vol_str = vol_str.upper().strip()
    last_char = vol_str[-1]

    if last_char in multiplier_dict.keys():
        multiplier_val = multiplier_dict[last_char]
        number_part = vol_str[:-1]
    else:
        multiplier_val = 1
        number_part = vol_str

    try:
        number = float(number_part.replace(',', ''))
    except ValueError:
        return np.nan

    return number * multiplier_val

def convert_change(change_str):
    if not isinstance(change_str, str):
        return change_str

    if change_str.endswith('%'):
        number_part = change_str[:-1]
    else:
        number_part = change_str

    try:
        number = float(number_part.replace(',', ''))
    except ValueError:
        return np.nan

    return number / 100.0

def load_and_process_data(file_path):
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    ## Xu ly du lieu thoi gian va dua Date thanh index
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)

    ## Chuyen doi ve dung dinh dang va dung don vi
    columns_convert_to_float = ["Price", "Open", "High", "Low"]
    for column in columns_convert_to_float:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace(",", "").astype(float)

    ## Bien doi Volume va Change
    df['Vol.'] = df['Vol.'].apply(convert_volume)
    df['Change %'] = df['Change %'].apply(convert_change)

    ## Doi ten cot ve dang chuan hon
    df.rename(columns={'Price': 'Close', 'Vol.': 'Volume', 'Change %': 'Change'}, inplace=True)

    ##Lay log cua Volume
    df['Volume'] = np.log1p(df['Volume'])

    return df.dropna()

def feature_extraction(df, create_labels=True):
    df = df.copy()
    df['SMA'] = ta.sma(df['Close'], length=20)
    df['EMA'] = ta.ema(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD_Hist'] = macd['MACDh_12_26_9']
    else:
        df['MACD_Hist'] = 0.0

    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None:
        df['BB_Width'] = bb['BBB_20_2.0_2.0']
        df['BB_Percentage'] = bb['BBP_20_2.0_2.0']
    
    if create_labels:
        df['Label'] = df['Close'].shift(-1)

    return df.dropna()

def split_data(df, train_ratio=0.7, val_ratio=0.2):
    ## Chia tap theo 70:20:10
    total = len(df)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size : train_size + val_size]
    test_data = df.iloc[train_size + val_size:]
    
    return train_data, val_data, test_data 

def scale_data(train_df, val_df, test_df, feature_cols):
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    ## Chi lay cac cot feature de scale
    train_features = train_df[feature_cols].values
    val_features = val_df[feature_cols].values
    test_features = test_df[feature_cols].values
    
    ## Chi fit scaler tren tap train
    scaler.fit(train_features)
    
    ## Transform o 3 tap
    train_scaled = scaler.transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)
    
    return scaler, train_scaled, val_scaled, test_scaled

def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :-1])
        y.append(data[i + window_size, -1])
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor

def train_model(model, train_loader, loss_func, optimizer, device):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_func(y_batch, y_pred)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

def validate_model(model, val_loader, loss_func, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch_v, y_batch_v in val_loader:
            X_batch_v, y_batch_v = X_batch_v.to(device), y_batch_v.to(device)

            y_pred_v = model(X_batch_v)
            loss = loss_func(y_batch_v, y_pred_v)

            val_loss += loss.item()

    return val_loss / len(val_loader)

def predict_model(model, test_loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            preds.extend(outputs.cpu().numpy())
            targets.extend(y_batch.numpy())
    
    return np.array(preds), np.array(targets)

def inverse_transform_target(scaled_data, scaler, target_index):
    ## Tao ma tran khong de lap day cac gia tri khong duoc du bao
    inverse_np = np.zeros((len(scaled_data), scaler.n_features_in_))

    inverse_np[:, target_index] = scaled_data.flatten()
    inversed = scaler.inverse_transform(inverse_np)

    return inversed[:, target_index]

def cal_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2
