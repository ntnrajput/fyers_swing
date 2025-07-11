from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

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
    df = df.copy()
    
    # 1. Price action features
    df['daily_return'] = df['close'].pct_change()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df['volatility_5'] = df['daily_return'].rolling(5).std()
    df['volatility_10'] = df['daily_return'].rolling(10).std()
    
    # 2. EMA relationship
    df['price_above_ema20'] = (df['close'] > df['ema20']).astype(int)
    df['price_above_ema50'] = (df['close'] > df['ema50']).astype(int)
    df['ema20_above_ema50'] = (df['ema20'] > df['ema50']).astype(int)

    # 3. Normalize dist to SR
    df['norm_dist_to_support'] = df['dist_to_support'] / df['close']
    df['norm_dist_to_resistance'] = df['dist_to_resistance'] / df['close']

    # 4. Handle initial NaNs
    df.fillna(0, inplace=True)

    return df


def train_swing_model(df_all):
    feature_columns = [
        'body', 'range', 'upper_shadow', 'lower_shadow',
        'daily_return', 'range_pct', 'volatility_5', 'volatility_10',
        'price_above_ema20', 'price_above_ema50', 'ema20_above_ema50',
        'norm_dist_to_support', 'norm_dist_to_resistance'
    ]
    
    X = df_all[feature_columns]
    y = df_all['target_hit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    print(classification_report(y_test, model.predict(X_test)))

    # Save model
    joblib.dump(model, "swing_model.pkl")
    print("✅ Model trained and saved as swing_model.pkl")

    return model


def predict_today(df_today, model):
    df_today = df_today.copy()
    add_candle_features(df_today)
    support_levels, resistance_levels = get_support_resistance(df_today, window=10)
    df_today = add_nearest_sr(df_today, support_levels, resistance_levels)
    df_today = add_model_features(df_today)

    feature_columns = [
        'body', 'range', 'upper_shadow', 'lower_shadow',
        'daily_return', 'range_pct', 'volatility_5', 'volatility_10',
        'price_above_ema20', 'price_above_ema50', 'ema20_above_ema50',
        'norm_dist_to_support', 'norm_dist_to_resistance'
    ]

    df_today['swing_prob'] = model.predict_proba(df_today[feature_columns])[:, 1]
    return df_today[df_today['swing_prob'] > 0.6]  # high-probability candidates
