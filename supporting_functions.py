import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bb_position(prices, period=20):
    """Calculate Bollinger Band position (0-1)"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position.clip(0, 1)

def add_candle_features(df):
    """Enhanced candle pattern features"""
    df = df.copy()
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_bearish'] = (df['open'] > df['close']).astype(int)
    
    # Additional candle features
    df['body_to_range'] = df['body'] / (df['range'] + 1e-8)  # Avoid division by zero
    df['upper_shadow_to_range'] = df['upper_shadow'] / (df['range'] + 1e-8)
    df['lower_shadow_to_range'] = df['lower_shadow'] / (df['range'] + 1e-8)
    
    # Doji pattern (small body relative to range)
    df['is_doji'] = (df['body_to_range'] < 0.1).astype(int)
    
    # Hammer pattern (small body, long lower shadow)
    df['is_hammer'] = ((df['body_to_range'] < 0.3) & 
                       (df['lower_shadow_to_range'] > 0.6)).astype(int)
    
    return df

def get_support_resistance(df, window=5):
    """Enhanced support/resistance detection with strength"""
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame must contain 'high' and 'low' columns")

    df = df.copy()
    
    # Rolling max/min (centered)
    rolling_max = df['high'].rolling(window=2*window+1, center=True).max()
    rolling_min = df['low'].rolling(window=2*window+1, center=True).min()

    df['Swing_High'] = (df['high'] == rolling_max)
    df['Swing_Low'] = (df['low'] == rolling_min)

    # Add strength based on how many times level is tested
    support_levels = []
    resistance_levels = []
    
    for idx, row in df[df['Swing_Low']].iterrows():
        # Count how many times this level is tested (within 1% range)
        level_price = row['low']
        strength = sum(abs(df['low'] - level_price) / level_price < 0.01)
        support_levels.append((idx, level_price, strength))
    
    for idx, row in df[df['Swing_High']].iterrows():
        level_price = row['high']
        strength = sum(abs(df['high'] - level_price) / level_price < 0.01)
        resistance_levels.append((idx, level_price, strength))

    return support_levels, resistance_levels

def add_nearest_sr(df, support_levels, resistance_levels):
    """Enhanced S/R features with strength"""
    support_data = [(level[1], level[2]) for level in support_levels]  # (price, strength)
    resistance_data = [(level[1], level[2]) for level in resistance_levels]

    nearest_support = []
    nearest_resistance = []
    support_strength = []
    resistance_strength = []

    for close in df['close']:
        if support_data:
            nearest_sup = min(support_data, key=lambda x: abs(x[0] - close))
            nearest_support.append(nearest_sup[0])
            support_strength.append(nearest_sup[1])
        else:
            nearest_support.append(close * 0.95)  # Default support 5% below
            support_strength.append(1)
        
        if resistance_data:
            nearest_res = min(resistance_data, key=lambda x: abs(x[0] - close))
            nearest_resistance.append(nearest_res[0])
            resistance_strength.append(nearest_res[1])
        else:
            nearest_resistance.append(close * 1.05)  # Default resistance 5% above
            resistance_strength.append(1)

    df['nearest_support'] = nearest_support
    df['nearest_resistance'] = nearest_resistance
    df['support_strength'] = support_strength
    df['resistance_strength'] = resistance_strength
    df['dist_to_support'] = df['close'] - df['nearest_support']
    df['dist_to_resistance'] = df['nearest_resistance'] - df['close']

    return df

def generate_swing_labels(df, target_pct=0.05, window=10, stop_loss_pct=0.03):
    """Enhanced labeling with stop loss consideration"""
    df = df.copy()
    target_hit = []
    max_return = []
    min_return = []

    for i in range(len(df)):
        if i + window >= len(df):
            target_hit.append(None)
            max_return.append(None)
            min_return.append(None)
        else:
            entry_price = df['close'].iloc[i]
            future_high = df['high'].iloc[i+1:i+1+window].max()
            future_low = df['low'].iloc[i+1:i+1+window].min()
            
            max_ret = (future_high - entry_price) / entry_price
            min_ret = (future_low - entry_price) / entry_price
            
            max_return.append(max_ret)
            min_return.append(min_ret)
            
            # Check if target hit before stop loss
            if max_ret >= target_pct and min_ret > -stop_loss_pct:
                target_hit.append(1)
            else:
                target_hit.append(0)

    df['target_hit'] = target_hit
    df['max_return'] = max_return
    df['min_return'] = min_return
    return df

def add_model_features(df):
    """Enhanced feature engineering"""
    df = df.copy()
    
    # 1. Price action features
    df['daily_return'] = df['close'].pct_change()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Multiple timeframe volatility
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['daily_return'].rolling(period).std()
        df[f'return_mean_{period}'] = df['daily_return'].rolling(period).mean()
    
    # 2. EMA relationship
    df['price_above_ema20'] = (df['close'] > df['ema20']).astype(int)
    df['price_above_ema50'] = (df['close'] > df['ema50']).astype(int)
    df['ema20_above_ema50'] = (df['ema20'] > df['ema50']).astype(int)
    
    # EMA distances (normalized)
    df['ema20_distance'] = (df['close'] - df['ema20']) / df['close']
    df['ema50_distance'] = (df['close'] - df['ema50']) / df['close']
    df['ema_spread'] = (df['ema20'] - df['ema50']) / df['close']

    # 3. Normalize dist to SR
    df['norm_dist_to_support'] = df['dist_to_support'] / df['close']
    df['norm_dist_to_resistance'] = df['dist_to_resistance'] / df['close']

    # 4. Volume features (if available)
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1)
        df['price_volume'] = df['close'] * df['volume']
    
    # 5. Momentum indicators
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['bb_position'] = calculate_bb_position(df['close'], 20)
    
    # 6. Trend strength
    df['trend_strength'] = df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # 7. Handle NaNs
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    return df

class TradingModelOptimizer:
    def __init__(self):
        self.best_params = {}
        self.results = []
        
    def optimize_parameters(self, df_all, param_grid=None):
        """Optimize model parameters using grid search"""
        if param_grid is None:
            param_grid = {
                'target_pct': [0.03, 0.05, 0.07, 0.10],
                'window': [10, 15, 20, 25],
                'stop_loss_pct': [0.02, 0.03, 0.04],
                'sr_window': [5, 10, 15, 20]
            }
        
        best_score = 0
        best_combination = None
        
        print("🔍 Starting parameter optimization...")
        
        # Get unique combinations
        import itertools
        keys = param_grid.keys()
        combinations = list(itertools.product(*param_grid.values()))
        
        for i, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            print(f"Testing combination {i+1}/{len(combinations)}: {params}")
            
            try:
                score = self._evaluate_parameters(df_all, params)
                self.results.append({**params, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_combination = params
                    
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        self.best_params = best_combination
        print(f"✅ Best parameters: {best_combination}")
        print(f"✅ Best score: {best_score:.4f}")
        
        return best_combination, best_score
    
    def _evaluate_parameters(self, df_all, params):
        """Evaluate a single parameter combination"""
        # Re-generate features with new parameters
        full_data = []
        
        for symbol in df_all['symbol'].unique():
            df = df_all[df_all['symbol'] == symbol].copy()
            
            # Apply new parameters
            support_levels, resistance_levels = get_support_resistance(df, window=params['sr_window'])
            df = add_nearest_sr(df, support_levels, resistance_levels)
            df = generate_swing_labels(df, 
                                     target_pct=params['target_pct'], 
                                     window=params['window'],
                                     stop_loss_pct=params['stop_loss_pct'])
            df = df.dropna(subset=['target_hit']).reset_index(drop=True)
            
            if len(df) > 0:
                full_data.append(df)
        
        if not full_data:
            return 0
            
        df_combined = pd.concat(full_data).reset_index(drop=True)
        
        # Train model with these parameters
        feature_columns = [
            'body', 'range', 'upper_shadow', 'lower_shadow', 'body_to_range',
            'daily_return', 'range_pct', 'volatility_5', 'volatility_10', 'volatility_20',
            'price_above_ema20', 'price_above_ema50', 'ema20_above_ema50',
            'norm_dist_to_support', 'norm_dist_to_resistance', 'support_strength',
            'resistance_strength', 'rsi', 'bb_position', 'trend_strength'
        ]
        
        # Filter features that exist
        available_features = [col for col in feature_columns if col in df_combined.columns]
        
        if len(available_features) == 0:
            return 0
            
        X = df_combined[available_features]
        y = df_combined['target_hit']
        
        if len(X) < 100 or y.sum() < 10:  # Need minimum data
            return 0
        
        # Use cross-validation for robust evaluation
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
        
        return scores.mean()
    
    def plot_results(self):
        """Plot optimization results"""
        if not self.results:
            print("No results to plot")
            return
            
        df_results = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot each parameter vs score
        params = ['target_pct', 'window', 'stop_loss_pct', 'sr_window']
        
        for i, param in enumerate(params):
            ax = axes[i//2, i%2]
            df_results.groupby(param)['score'].mean().plot(kind='bar', ax=ax)
            ax.set_title(f'{param} vs Score')
            ax.set_xlabel(param)
            ax.set_ylabel('Average Score')
        
        plt.tight_layout()
        plt.show()

def train_enhanced_model(df_all, model_params=None):
    """Train enhanced model with hyperparameter tuning"""
    feature_columns = [
        'body', 'range', 'upper_shadow', 'lower_shadow', 'body_to_range',
        'upper_shadow_to_range', 'lower_shadow_to_range', 'is_doji', 'is_hammer',
        'daily_return', 'range_pct', 'volatility_5', 'volatility_10', 'volatility_20',
        'return_mean_5', 'return_mean_10', 'return_mean_20',
        'price_above_ema20', 'price_above_ema50', 'ema20_above_ema50',
        'ema20_distance', 'ema50_distance', 'ema_spread',
        'norm_dist_to_support', 'norm_dist_to_resistance', 
        'support_strength', 'resistance_strength',
        'rsi', 'bb_position', 'trend_strength'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in df_all.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    X = df_all[available_features]
    y = df_all['target_hit']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )
    
    # Hyperparameter tuning
    if model_params is None:
        model_params = {
            'n_estimators': [100],    #[100,200,300]
            'max_depth': [10, None],  #  [10,20,None]
            'min_samples_split': [2, 5],   # [2,5,10]
            'min_samples_leaf': [1,]  # [1,2,4]
        }
    
    print("🔍 Performing hyperparameter tuning...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, model_params, cv=3, scoring='roc_auc', n_jobs=-1)  # cv=5
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Best parameters: {grid_search.best_params_}")
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Save model and scaler
    joblib.dump(best_model, "enhanced_swing_model.pkl")
    joblib.dump(scaler, "feature_scaler.pkl")
    joblib.dump(available_features, "feature_columns.pkl")
    
    print("✅ Enhanced model saved!")
    
    return best_model, scaler, available_features

def backtest_model(df_all, model, scaler, feature_columns, start_date=None, end_date=None):
    """Backtest the model on historical data"""
    print("🔄 Running backtest...")
    
    # Convert timestamp column to datetime if it's not already
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    
    if start_date:
        df_all = df_all[df_all['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df_all = df_all[df_all['timestamp'] <= pd.to_datetime(end_date)]
    
    results = []
    
    for symbol in df_all['symbol'].unique():
        df_symbol = df_all[df_all['symbol'] == symbol].copy()
        
        try:
            # Ensure we have all required features
            df_symbol = add_candle_features(df_symbol)
            support_levels, resistance_levels = get_support_resistance(df_symbol, window=10)
            df_symbol = add_nearest_sr(df_symbol, support_levels, resistance_levels)
            df_symbol = add_model_features(df_symbol)
            
            # Get predictions (only if we have all required features)
            available_features = [col for col in feature_columns if col in df_symbol.columns]
            if not available_features:
                continue
                
            X = scaler.transform(df_symbol[available_features])
            df_symbol['prediction'] = model.predict_proba(X)[:, 1]
            
            # Simulate trades (buy when probability > 0.6)
            trades = df_symbol[df_symbol['prediction'] > 0.6].copy()
            
            for _, trade in trades.iterrows():
                if pd.notna(trade['max_return']):
                    results.append({
                        'symbol': symbol,
                        'entry_date': trade['timestamp'].strftime('%Y-%m-%d'),
                        'entry_price': trade['close'],
                        'prediction': trade['prediction'],
                        'actual_return': trade['max_return'],
                        'target_hit': trade['target_hit']
                    })
        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e}")
            continue
    
    if results:
        backtest_df = pd.DataFrame(results)
        # Convert entry_date to datetime for sorting
        backtest_df['entry_date'] = pd.to_datetime(backtest_df['entry_date'])
        backtest_df = backtest_df.sort_values('entry_date')
        
        print(f"\n📊 Backtest Results ({len(backtest_df)} trades):")
        print(f"Win Rate: {backtest_df['target_hit'].mean():.2%}")
        print(f"Average Return: {backtest_df['actual_return'].mean():.2%}")
        print(f"Average Prediction Score: {backtest_df['prediction'].mean():.3f}")
        
        # Calculate cumulative returns
        backtest_df['cumulative_return'] = (1 + backtest_df['actual_return']).cumprod() - 1
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_df['entry_date'], backtest_df['cumulative_return'])
        plt.title('Cumulative Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.show()
        
        return backtest_df
    else:
        print("❌ No valid trades found in backtest period")
        return pd.DataFrame()

def predict_today(df_today, model, scaler, feature_columns, probability_threshold=0.6):
    df_today = df_today.copy()
    df_today = add_candle_features(df_today)
    support_levels, resistance_levels = get_support_resistance(df_today, window=10)
    df_today = add_nearest_sr(df_today, support_levels, resistance_levels)
    df_today = add_model_features(df_today)

    available_features = [col for col in feature_columns if col in df_today.columns]
    
    if not available_features:
        raise ValueError("No matching features between training and prediction data")
    
    # Scale features and predict
    X = scaler.transform(df_today[available_features])
    df_today['swing_prob'] = model.predict_proba(X)[:, 1]
    
    # Return high-probability candidates with all relevant info
    return df_today[df_today['swing_prob'] > probability_threshold][[
        'symbol', 'timestamp', 'close', 'swing_prob', 
        'rsi', 'bb_position', 'nearest_support', 
        'nearest_resistance', 'ema20', 'ema50'
    ]].sort_values('swing_prob', ascending=False)

# Example usage:
if __name__ == "__main__":
    # Load your data
    # df_all = your_data_loading_function()
    
    # 1. Optimize parameters
    optimizer = TradingModelOptimizer()
    # best_params, best_score = optimizer.optimize_parameters(df_all)
    # optimizer.plot_results()
    
    # 2. Train enhanced model
    # model, scaler, features = train_enhanced_model(df_all)
    
    # 3. Run backtest
    # backtest_results = backtest_model(df_all, model, scaler, features)
    
    print("🎯 Trading model optimization complete!")