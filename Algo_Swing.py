from fyers_apiv3 import fyersModel
from datetime import datetime, timezone, timedelta
import os
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from side_functions import get_support_levels
from concurrent.futures import ThreadPoolExecutor, as_completed
from side_functions import get_history


# Configuration
client_id = "VE3CCLJZWA-100" 
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb2FUckVsVjQ2OTJBUVR0dnhTeDBTZzZNODhuN1VJbVF1TlFHYUljRHlYQ0RSRFJEXy1Ed2NWMFB6TjVYR2NfNmJNMDJXWDdsblh4cmlmY0VnWlU1TExvejYxTTNrZEhIbHJMTVZMRmw1cXRTS282Zz0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5YjViNjVmY2VmMzliNjJjZDlkZjBjZmU4YzhjYmRlMDk3ZDQxYmRkMGRlMmFiNWZlZjgwYWZjYyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWFQwMjYyNCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzUxNzYxODAwLCJpYXQiOjE3NTE3MjY3ODgsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc1MTcyNjc4OCwic3ViIjoiYWNjZXNzX3Rva2VuIn0.QnqlHpUF5l2PfRziFNk0v0BQO3IGx6GDVuUSbVJToac"
fyers = fyersModel.FyersModel(client_id = client_id, is_async=False, token = access_token, log_path="")
today = datetime.today().date()
nifty_history =f"nifty_history.csv"


def update_all_symbol_history():
    df_symbol = pd.read_csv("all_symbols_history.csv", usecols =['timestamp','open','high','low','close','volume','symbol','ema20','ema50'])
    last_date = pd.to_datetime(df_symbol['timestamp'].iloc[-1]).date()  + timedelta(days=1) 
    symbols =df_symbol['symbol'].unique().tolist()
    today = datetime.today().date()
    df_updated = pd.DataFrame()

    symbol='NSE:RELIANCE-EQ'

    for symbol in symbols:
        data = {"symbol": symbol, "resolution": "1D", "date_format": "1", "range_from": last_date, "range_to": today, "cont_flag": "1"}
        res = fyers.history(data=data)
        df_old = df_symbol[df_symbol['symbol']==symbol]
        df_new = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df_new ['symbol'] = symbol
        close_prices = pd.concat([df_old['close'],df_new['close']]).tolist()
        df_new['ema20'] = sum(close_prices[-20:])/20
        df_new['ema50'] = sum(close_prices[-50:])/50
        df_updated = pd.concat([df_updated, df_old, df_new])
        df_updated = df_updated.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')

        print(df_updated)




        

def check_trade():
    df_symbol = pd.read_csv("all_symbols_history.csv", usecols =['timestamp','open','high','low','close','volume','symbol','ema20','ema50'])

    symbols = (df_symbol['symbol'].unique().tolist())
    for symbol in symbols:
        df = df_symbol[df_symbol['symbol']==symbol]
        volume_avg = df ['volume'].mean()
        if volume_avg < 1000000:
            continue

update_all_symbol_history()
    
    


# print(df_symbol[df_symbol['symbol']=='NSE:RELIANCE-EQ'])





















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
