from fyers_apiv3 import fyersModel
from datetime import datetime, timezone, timedelta
import os
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from side_functions import get_support_levels
from concurrent.futures import ThreadPoolExecutor, as_completed
from supporting_functions import add_candle_features, get_support_resistance,add_nearest_sr, generate_swing_labels, add_model_features, train_swing_model, predict_today
import joblib

# Configuration
client_id = "VE3CCLJZWA-100" 
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb2FsaS1zclFWSWpqZWJPR01UYXFMSnhTWks3TkpYRnhFUWJ1aF96dlU4d1I5S1VrZ09XWnpkcjBJdlhneXBrYjhUUXBLRndkMVhHcm84TS10RWFhYVlTbENXUC1ZM0tHQVlwYkh0RVcwbHVqeDZPOD0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5YjViNjVmY2VmMzliNjJjZDlkZjBjZmU4YzhjYmRlMDk3ZDQxYmRkMGRlMmFiNWZlZjgwYWZjYyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWFQwMjYyNCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzUxODQ4MjAwLCJpYXQiOjE3NTE3OTk5OTgsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc1MTc5OTk5OCwic3ViIjoiYWNjZXNzX3Rva2VuIn0.1MdOvpEpqt6A4xOPHTkdU0turL-PNyEhVH73WtaexYU"

fyers = fyersModel.FyersModel(client_id = client_id, is_async=False, token = access_token, log_path="")
today = datetime.today().date()
nifty_history =f"nifty_history.csv"


def update_all_symbol_history():
    df_symbol = pd.read_csv("all_symbols_history.csv", usecols =['timestamp','open','high','low','close','volume','symbol','ema20','ema50'])
    # last_date = pd.to_datetime(df_symbol['timestamp'].iloc[-1]).date()  + timedelta(days=1) 
    last_date = pd.to_datetime(df_symbol['timestamp'].iloc[-1]).date() 
    symbols =df_symbol['symbol'].unique().tolist()
    today = datetime.today().date()
    df_updated = pd.DataFrame()

    # symbol='NSE:RELIANCE-EQ'

    for symbol in symbols:
        df_old = df_symbol[df_symbol['symbol']==symbol].iloc[:-1]
        data = {"symbol": symbol, "resolution": "1D", "date_format": "1", "range_from": last_date, "range_to": today, "cont_flag": "1"}
        res = fyers.history(data=data)

        if res.get('candles'):
            df_new = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            df_new ['symbol'] = symbol
    

        close_prices = pd.concat([df_old['close'],df_new['close']]).tolist()
        df_new['ema20'] = sum(close_prices[-20:])/20
        df_new['ema50'] = sum(close_prices[-50:])/50
        df_updated = pd.concat([df_updated, df_old, df_new])
        df_updated = df_updated.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')

    df_updated.to_csv('all_symbols_history.csv', index=False)




def check_trade():
    df_symbol = pd.read_csv("all_symbols_history.csv", usecols =['timestamp','open','high','low','close','volume','symbol','ema20','ema50'])
    symbols = (df_symbol['symbol'].unique().tolist())[300:325]
    
    full_data = []

    for symbol in symbols:
        df = df_symbol[df_symbol['symbol']==symbol].copy()
        volume_avg = df ['volume'].iloc[-50:].mean()
        price_avg = df['close'].iloc[-50].mean()
        print(symbol, volume_avg)
        if  volume_avg < 5e5 and price_avg < 50:
            continue
        add_candle_features(df)
        support_levels, resistance_levels = get_support_resistance(df, window=10)
        df = add_nearest_sr(df, support_levels, resistance_levels)
        df = generate_swing_labels(df, target_pct=0.05, window=10)
        df = df.dropna(subset=['target_hit']).reset_index(drop=True)
        df = add_model_features(df) 
        full_data.append(df)

        df.to_csv('df_print.csv', index=False)

    df_all = pd.concat(full_data).reset_index(drop=True)
    return df_all
        

        
df_all = check_trade()
model = train_swing_model(df_all)


model = joblib.load("swing_model.pkl")
df_symbol = pd.read_csv("all_symbols_history.csv", usecols =['timestamp','open','high','low','close','volume','symbol','ema20','ema50'])
   
df_today = df_symbol.sort_values('timestamp').groupby('symbol').tail(1)

df_candidates = predict_today(df_today, model)
print(df_candidates[['symbol', 'close', 'swing_prob']])

    





















# pos = fyers.positions()
# pd.DataFrame(pos.get('netPositions', [])).to_csv(csv_file, index=False)



# get_history('NSE:NIFTY50-INDEX', '1D', 31536000, nifty_history)
# get_support_levels()

# def has_open_positions():
#     try:
#         df = pd.read_csv(csv_file, usecols=['symbol', 'buyQty', 'sellQty'])
#         df['net'] = df['buyQty'] - df['sellQty']
#         return not df[df['net'] > 0].empty
#     except: return False

# def search_trade():
#     get_history('NSE:Nifty50-INDEX')
#     ema_logic = check_ema_logic()
#     print(ema_logic)
    
# def check_ema_logic():
#     df = pd.read_csv(history_csv)
#     ema_20 = df['ema20'].iloc[-1]
#     ema_50 = df['ema50'].iloc[-1]
#     if ema_20 > ema_50:
#         return "Bullish"
#     else:
#         return "Bearish"

# def monitor():
#     global last_check
#     now = datetime.now()
#     if MARKET_START <= now.time() <= MARKET_END:
#         if has_open_positions():
#             get_history('NSE:Nifty50-INDEX')
#         elif now.strftime('%H:%M') != last_check:
#             search_trade()
#             last_check = now.strftime('%H:%M')

# # while True:
#     monitor()    
#     time.sleep(CHECK_INTERVAL)
