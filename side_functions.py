# from fyers_apiv3 import fyersModel
# from datetime import datetime, timezone, timedelta
# import os
import csv
# import time
import pandas as pd
import matplotlib.pyplot as plt
# from Algo_1 import get_history

nifty_history =f"nifty_history.csv"
levels = f"nifty_levels.csv"

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
    

if __name__ == "__main__":
    get_support_levels()