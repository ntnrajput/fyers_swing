from fyers_apiv3 import fyersModel
from datetime import datetime, timezone, timedelta
import os
import csv
import time
import pandas as pd

# Configuration
client_id = "VE3CCLJZWA-100" 
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb1hfdm82UDNuWkE5bzdhMlRoVUkxYU51eEtJNUo1NEFIVzZGQ3NvLUdtR2pHZDQxbG1nR1Z3OVhaUDB6Z1haLVV1YkhGaVdQUkNZT1JzbHpYTVMxNjZYcXN5U01Nc0tRaVRacE41R2pIanNrcTJ3az0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5YjViNjVmY2VmMzliNjJjZDlkZjBjZmU4YzhjYmRlMDk3ZDQxYmRkMGRlMmFiNWZlZjgwYWZjYyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWFQwMjYyNCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzUxMTU3MDAwLCJpYXQiOjE3NTExMjA4NzIsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc1MTEyMDg3Miwic3ViIjoiYWNjZXNzX3Rva2VuIn0.RNYwnDVl3dG-Q4yXl4tmc4VUGYPFTnphuLZS3JFfc94"
fyers = fyersModel.FyersModel(client_id = client_id, is_async=False, token = access_token, log_path="")
today = datetime.today().date()
csv_file = f"{today}.csv"
history_csv = f"history_{today}.csv"
nifty_history =f"nifty_history.csv"
CHECK_INTERVAL = 60
MARKET_START, MARKET_END = datetime.strptime("09:15", "%H:%M").time(), datetime.strptime("23:30", "%H:%M").time()
last_check = ""

pos = fyers.positions()
pd.DataFrame(pos.get('netPositions', [])).to_csv(csv_file, index=False)


def get_history(symbol, resolution ='5', from_sec = 300, rec_file = history_csv):
    now = int(time.time())
    past = now - from_sec
    data = {"symbol": symbol, "resolution": resolution, "date_format": "0", "range_from": str(past), "range_to": str(now), "cont_flag": "1"}
    try:
        res = fyers.history(data=data)
        if res.get('candles'):
            df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            df['symbol'] = symbol
            if os.path.exists(rec_file):
                old = pd.read_csv(rec_file)
                old['timestamp'] = pd.to_datetime(old['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
                df = pd.concat([old, df]).drop_duplicates(subset=['timestamp', 'symbol'], keep='last')

            df.sort_values('timestamp', inplace=True)
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

            df.to_csv(rec_file, index=False)
            
    except Exception as e:
        print(f"History error: {e}")

def get_support_levels():
    nifty_history = get_history('NSE:NIFTY50-INDEX', '1D', 31536000, 'nifty_history')
    

get_history('NSE:Nifty50-INDEX','5', 259200)
get_support_levels()

def has_open_positions():
    try:
        df = pd.read_csv(csv_file, usecols=['symbol', 'buyQty', 'sellQty'])
        df['net'] = df['buyQty'] - df['sellQty']
        return not df[df['net'] > 0].empty
    except: return False

def search_trade():
    get_history('NSE:Nifty50-INDEX')
    ema_logic = check_ema_logic()
    print(ema_logic)
    
def check_ema_logic():
    df = pd.read_csv(history_csv)
    ema_20 = df['ema20'].iloc[-1]
    ema_50 = df['ema50'].iloc[-1]
    if ema_20 > ema_50:
        return "Bullish"
    else:
        return "Bearish"

def monitor():
    global last_check
    now = datetime.now()
    if MARKET_START <= now.time() <= MARKET_END:
        if has_open_positions():
            get_history('NSE:Nifty50-INDEX')
        elif now.strftime('%H:%M') != last_check:
            search_trade()
            last_check = now.strftime('%H:%M')

while True:
    monitor()    
    time.sleep(CHECK_INTERVAL)
