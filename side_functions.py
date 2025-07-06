from fyers_apiv3 import fyersModel
# from datetime import datetime, timezone, timedelta
import os
import csv
import time
# import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
# from Algo_1 import get_history


client_id = "VE3CCLJZWA-100" 
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJWRTNDQ0xKWldBIiwidXVpZCI6IjUwN2ZlMjVkMjRhOTQxNDJiYTZmYjM3YjI5NDQ0YTgwIiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IlhUMDI2MjQiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5YjViNjVmY2VmMzliNjJjZDlkZjBjZmU4YzhjYmRlMDk3ZDQxYmRkMGRlMmFiNWZlZjgwYWZjYyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImF1ZCI6IltcImQ6MVwiLFwiZDoyXCIsXCJ4OjBcIixcIng6MVwiLFwieDoyXCJdIiwiZXhwIjoxNzUxODI5NDczLCJpYXQiOjE3NTE3OTk0NzMsImlzcyI6ImFwaS5sb2dpbi5meWVycy5pbiIsIm5iZiI6MTc1MTc5OTQ3Mywic3ViIjoiYXV0aF9jb2RlIn0.jaruS7A6sb-HbID1Kh-UZ5SLUakDZb8kQf1butKHivs"
fyers = fyersModel.FyersModel(client_id = client_id, is_async=False, token = access_token, log_path="")


nifty_history =f"nifty_history.csv"
levels = f"nifty_levels.csv"
history_csv = f"symbol_history.csv"


def get_support_levels():
    
    df = pd.read_csv(nifty_history, usecols =['timestamp','close'])
    # print(df)

    df.set_index('timestamp', inplace=True)

    # Step 1: Find Local Highs and Lows
    window = 15
    df['local_max'] = df['close'][(df['close'] == df['close'].rolling(window, center=True).max())]
    df['local_min'] = df['close'][(df['close'] == df['close'].rolling(window, center=True).min())]

    highs = df[df['local_max'].notna()]
    lows = df[df['local_min'].notna()]

    # Step 2: Cluster Nearby Levels
    def cluster_levels(levels, threshold=0.5):
        clustered = []
        for level in sorted(levels):
            if not clustered:
                clustered.append(level)
            elif abs(level - clustered[-1]) / clustered[-1] > threshold / 100:
                clustered.append(level)
        return clustered

    resistance_levels = cluster_levels(highs['close'].values, threshold=0.5)
    support_levels = cluster_levels(lows['close'].values, threshold=0.5)
    combined_levels = resistance_levels+support_levels
    combined_levels.sort()
    df_levels = pd.DataFrame(combined_levels,columns=['levels'])

    df_levels.to_csv(levels, index=False)


    # Step 3: Plot the Close Price with Levels
    # plt.figure(figsize=(14, 6))
    # plt.plot(df['close'], label='Nifty Close', color='blue')

    # for level in resistance_levels:
    #     plt.axhline(level, color='red', linestyle='--', alpha=0.5, label='Resistance' if level == resistance_levels[0] else "")
    # for level in support_levels:
    #     plt.axhline(level, color='green', linestyle='--', alpha=0.5, label='Support' if level == support_levels[0] else "")

    # plt.title("Nifty Support and Resistance Levels")
    # plt.xlabel("Date")
    # plt.ylabel("Close Price")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


def get_history(symbol, resolution ='1D', from_sec = 86400*30, rec_file = history_csv,i=0):
    now = int(time.time()) -(86400*30*12*i)
    past = now - from_sec
    data = {"symbol": symbol, "resolution": resolution, "date_format": "0", "range_from": str(past), "range_to": str(now), "cont_flag": "1"}

    try:
        res = fyers.history(data=data)
        print(f"Received candles: {len(res.get('candles', []))}")

        if res.get('candles'):
            df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            df['symbol'] = symbol
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            return df


            # if os.path.exists(rec_file):
            #     old = pd.read_csv(rec_file)
            #     old['timestamp'] = pd.to_datetime(old['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
            #     df = pd.concat([old, df]).drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        

            # df.sort_values(by=['symbol', 'timestamp'], inplace=True)  # <-- Add this
            # df.to_csv(rec_file, index=False)
            
    except Exception as e:
        print(f"History error: {e}")

    return pd.DataFrame()
    

if __name__ == "__main__":
    get_support_levels()