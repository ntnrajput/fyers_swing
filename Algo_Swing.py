from fyers_apiv3 import fyersModel
from datetime import datetime, timezone, timedelta
import os
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from side_functions import get_support_levels
from concurrent.futures import ThreadPoolExecutor, as_completed

from supporting_functions import (
    add_candle_features, get_support_resistance, add_nearest_sr, 
    generate_swing_labels, add_model_features_enhanced, train_enhanced_model_with_volume, 
    predict_today_with_filters, TradingModelOptimizer, backtest_model, calculate_rsi, 
    calculate_bb_position, filter_quality_stocks, add_enhanced_volume_features
)

import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
client_id = "VE3CCLJZWA-100" 
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb2NJcVhFc0Q2ZFRxWE0zVFhTMkIyWnpJa0djY0pFdUVOUnRtV080ekhrRW5pM0lhSDEwX3lENDQ2LWxSWDZtekhJeFJUODRneTNxaVBnTzVxNjBWWGVQMk5uS05jVlBkcHh1ZmZZUnJvNlNsMFNMND0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI5YjViNjVmY2VmMzliNjJjZDlkZjBjZmU4YzhjYmRlMDk3ZDQxYmRkMGRlMmFiNWZlZjgwYWZjYyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWFQwMjYyNCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzUyMjgwMjAwLCJpYXQiOjE3NTIyMDU5NzUsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc1MjIwNTk3NSwic3ViIjoiYWNjZXNzX3Rva2VuIn0.8DFmsxKb9bwqFp2HcHZpqA7GE3vZsbplhL4Z24CU0jU"

fyers = fyersModel.FyersModel(client_id = client_id, is_async=False, token = access_token, log_path="")
today = datetime.today().date()
nifty_history =f"nifty_history.csv"

def update_all_symbol_history():
    """Enhanced history update with error handling"""
    try:
        # df_symbol = pd.read_csv("all_symbols_history.csv", 
        #                        usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'ema20', 'ema50'])
        df_symbol = pd.read_parquet("all_symbols_history.parquet", 
                             columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'ema20', 'ema50'])
        
        last_date = pd.to_datetime(df_symbol['timestamp'].iloc[-1]).date()
        symbols = df_symbol['symbol'].unique().tolist()
        today = datetime.today().date()
        df_updated = pd.DataFrame()
        
        print(f"📊 Updating {len(symbols)} symbols from {last_date} to {today}")
        
        for i, symbol in enumerate(symbols):
            print(f"Processing {i+1}/{len(symbols)}: {symbol}")
            
            try:
                df_old = df_symbol[df_symbol['symbol'] == symbol].iloc[:-1]
                data = {
                    "symbol": symbol, 
                    "resolution": "1D", 
                    "date_format": "1", 
                    "range_from": last_date, 
                    "range_to": today, 
                    "cont_flag": "1"
                }
                res = fyers.history(data=data)

                if res.get('candles'):
                    df_new = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
                    df_new['symbol'] = symbol

                    # Enhanced EMA calculation
                    close_prices = pd.concat([df_old['close'], df_new['close']]).tolist()
                    if len(close_prices) >= 20:
                        df_new['ema20'] = sum(close_prices[-20:]) / 20
                    else:
                        df_new['ema20'] = df_new['close'].mean()
                        
                    if len(close_prices) >= 50:
                        df_new['ema50'] = sum(close_prices[-50:]) / 50
                    else:
                        df_new['ema50'] = df_new['close'].mean()
                    
                    df_updated = pd.concat([df_updated, df_old, df_new])
                else:
                    # No new data, keep old data
                    df_updated = pd.concat([df_updated, df_old])
                    
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                # Keep old data even if update fails
                df_old = df_symbol[df_symbol['symbol'] == symbol]
                df_updated = pd.concat([df_updated, df_old])
                continue
        
        # Remove duplicates and save
        df_updated = df_updated.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')

        df_updated['timestamp'] = pd.to_datetime(df_updated['timestamp'], utc=True, errors='coerce')
        df_updated = df_updated.dropna(subset=['timestamp'])

        df_updated.to_csv('all_symbols_history.csv', index=False)
        df_updated.to_parquet("all_symbols_history.parquet", compression='snappy')

        print("✅ History update completed successfully!")
        
    except Exception as e:
        print(f"❌ Critical error in update_all_symbol_history: {e}")
        raise

def check_trade_enhanced(optimize_params=False, use_saved_model=True):
    """Enhanced trade checking with volume analysis and filtering"""
    try:
        df_symbol = pd.read_csv("all_symbols_history.csv", 
                               usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'ema20', 'ema50'])
        symbols = df_symbol['symbol'].unique().tolist()
        
        print(f"🔍 Analyzing {len(symbols)} symbols with enhanced volume analysis...")
        
        full_data = []
        processed_symbols = 0
        filtered_symbols = 0

        for symbol in symbols:
            try:
                df = df_symbol[df_symbol['symbol'] == symbol].copy()
                
                # Apply quality filters first
                if not filter_quality_stocks(df, min_price=25, min_volume=100000):
                    continue
                    
                filtered_symbols += 1
                print(f"✅ {symbol} passed quality filters ({filtered_symbols}/{len(symbols)})")
                
                # Enhanced processing with volume features
                df = add_candle_features(df)
                support_levels, resistance_levels = get_support_resistance(df, window=20)
                df = add_nearest_sr(df, support_levels, resistance_levels)
                df = generate_swing_labels(df, target_pct=0.05, window=20)
                df = df.dropna(subset=['target_hit']).reset_index(drop=True)
                
                if len(df) < 20:
                    continue
                    
                # Use enhanced feature engineering
                df = add_model_features_enhanced(df)
                full_data.append(df)
                processed_symbols += 1
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                continue

        if not full_data:
            print("❌ No valid data found for any symbol")
            return pd.DataFrame()
            
        print(f"✅ Successfully processed {processed_symbols} high-quality symbols out of {filtered_symbols} filtered symbols")
        df_all = pd.concat(full_data, ignore_index=True)
        
        # Save processed data
        df_all.to_csv('processed_data_enhanced.csv', index=False)
        
        return df_all
        
    except Exception as e:
        print(f"❌ Critical error in check_trade_enhanced: {e}")
        raise

def train_and_save_enhanced_model(df_all):
    """Train enhanced model with volume features"""
    try:
        print("🤖 Training enhanced model with volume analysis...")
        model, scaler, features = train_enhanced_model_with_volume(df_all)
        
        # Save additional info
        model_info = {
            'train_date': datetime.now().isoformat(),
            'n_samples': len(df_all),
            'n_features': len(features),
            'features': features,
            'model_type': 'enhanced_with_volume',
            'filters_applied': {
                'min_price': 25,
                'min_volume': 100000
            }
        }
        
        import json
        with open('enhanced_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
            
        print("✅ Enhanced model training completed and saved!")
        return model, scaler, features
        
    except Exception as e:
        print(f"❌ Error in enhanced model training: {e}")
        raise

def get_enhanced_trading_candidates(probability_threshold=0.6, min_price=5, min_volume=100000):
    """Get today's trading candidates with enhanced filtering"""
    try:
        # Load saved enhanced model
        model = joblib.load("enhanced_swing_model_with_volume.pkl")
        scaler = joblib.load("feature_scaler_with_volume.pkl")
        features = joblib.load("feature_columns_with_volume.pkl")
        
        print("✅ Enhanced model loaded successfully")
        
        # Get today's data
        df_symbol = pd.read_csv("all_symbols_history.csv", 
                               usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'ema20', 'ema50'])
        
        # Get latest data for each symbol
        df_today = df_symbol.sort_values('timestamp').groupby('symbol').tail(1).copy()
        
        print(f"📊 Analyzing {len(df_today)} symbols with enhanced filtering...")
        
        # Get filtered predictions
        df_candidates = predict_today_with_filters(
            df_today, model, scaler, features, 
            probability_threshold=probability_threshold,
            min_price=min_price, 
            min_volume=min_volume
        )
        
        if len(df_candidates) > 0:
            print(f"🎯 Found {len(df_candidates)} high-quality candidates:")
            
            # Display enhanced candidate info
            display_cols = ['symbol', 'close', 'swing_prob', 'rsi', 'bb_position', 
                          'volume_ratio_20', 'volume_spike', 'volume_confirmed_trend']
            available_display_cols = [col for col in display_cols if col in df_candidates.columns]
            
            print(df_candidates[available_display_cols].round(3))
            
            # Save candidates with timestamp
            df_candidates.to_csv('enhanced_candidates.csv', index=False)
            
            return df_candidates
        else:
            print("❌ No high-quality candidates found today")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error getting enhanced trading candidates: {e}")
        return pd.DataFrame()



def run_backtest(months_back=6):
    """Run backtest on historical data"""
    try:
        # Load model
        model = joblib.load("enhanced_swing_model.pkl")
        scaler = joblib.load("feature_scaler.pkl")
        features = joblib.load("feature_columns.pkl")
        
        # Load data
        df_all = pd.read_csv('processed_data.csv')
        
        # Define backtest period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back*30)
        
        print(f"📈 Running backtest from {start_date.date()} to {end_date.date()}")
        
        # Run backtest
        backtest_results = backtest_model(df_all, model, scaler, features, 
                                        start_date=start_date, end_date=end_date)
        
        if len(backtest_results) > 0:
            backtest_results.to_csv('backtest_results.csv', index=False)
            print("✅ Backtest completed and saved!")
        
        return backtest_results
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return pd.DataFrame()


def main_enhanced():
    """Enhanced main execution function"""
    print("🚀 Starting Enhanced Trading System with Volume Analysis")
    print("=" * 60)
    
    try:
        # Step 1: Update historical data
        print("📊 Step 1: Updating historical data...")
        # update_all_symbol_history()
        
        # Step 2: Enhanced processing with volume analysis and filtering
        print("\n🔍 Step 2: Enhanced processing with volume analysis...")
        df_all = check_trade_enhanced(optimize_params=False)
        
        if df_all.empty:
            print("❌ No quality data to process. Exiting.")
            return
        
        # Step 3: Train enhanced model
        print("\n🤖 Step 3: Training enhanced model with volume features...")
        model, scaler, features = train_and_save_enhanced_model(df_all)
        
        # Step 4: Get enhanced trading candidates
        print("\n🎯 Step 4: Getting enhanced trading candidates...")
        candidates = get_enhanced_trading_candidates(
            probability_threshold=0.6, 
            min_price=25, 
            min_volume=100000
        )
        
        # Step 5: Run backtest (use the fixed backtest function)
        print("\n📈 Step 5: Running backtest...")
        backtest_results = run_backtest(months_back=6)
        
        print("\n✅ Enhanced trading system execution completed!")
        print("=" * 60)
        
        # Enhanced Summary
        print("\n📋 ENHANCED SUMMARY:")
        print(f"• Total symbols analyzed: {len(df_all['symbol'].unique()) if not df_all.empty else 0}")
        print(f"• High-quality candidates: {len(candidates) if not candidates.empty else 0}")
        print(f"• Backtest trades: {len(backtest_results) if not backtest_results.empty else 0}")
        print(f"• Volume analysis: ✅ Enabled")
        print(f"• Quality filters: Price >= ₹25, Volume >= 1L")
        
        if not candidates.empty:
            top_candidate = candidates.iloc[0]
            print(f"• Top candidate: {top_candidate['symbol']} (prob: {top_candidate['swing_prob']:.3f})")
            if 'volume_ratio_20' in top_candidate:
                print(f"  - Volume ratio: {top_candidate['volume_ratio_20']:.2f}x")
            if 'volume_confirmed_trend' in top_candidate:
                print(f"  - Volume confirmed trend: {'Yes' if top_candidate['volume_confirmed_trend'] else 'No'}")
            
    except Exception as e:
        print(f"❌ Critical error in enhanced main execution: {e}")
        raise

if __name__ == "__main__":
    main_enhanced()

# Alternative: Run specific functions individually
# 
# # Just update data
# update_all_symbol_history()
# 
# # Just get candidates (if model already exists)
# candidates = get_trading_candidates()
# 
# # Just run backtest
# backtest_results = run_backtest()