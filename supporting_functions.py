from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def add_candle_features(df):
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_bearish'] = (df['open'] > df['close']).astype(int)

def get_support_resistance(df, window=5):
    """
    Detects support and resistance levels using rolling windows on high and low.

    Args:
        df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
        window (int): Half-window size (total window will be 2*window + 1).

    Returns:
        support_levels (list): List of (timestamp, low) tuples as support.
        resistance_levels (list): List of (timestamp, high) tuples as resistance.
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame must contain 'high' and 'low' columns")

    df = df.copy()  # to avoid modifying original

    # Rolling max/min (centered)
    rolling_max = df['high'].rolling(window=2*window+1, center=True).max()
    rolling_min = df['low'].rolling(window=2*window+1, center=True).min()

    df['Swing_High'] = (df['high'] == rolling_max)
    df['Swing_Low'] = (df['low'] == rolling_min)

    support_levels = [(idx, row['low']) for idx, row in df[df['Swing_Low']].iterrows()]
    resistance_levels = [(idx, row['high']) for idx, row in df[df['Swing_High']].iterrows()]

    return support_levels, resistance_levels


def add_nearest_sr(df, support_levels, resistance_levels):
    support_prices = [level[1] for level in support_levels]
    resistance_prices = [level[1] for level in resistance_levels]

    nearest_support = []
    nearest_resistance = []

    for close in df['close']:
        if support_prices:
            nearest_support.append(min(support_prices, key=lambda x: abs(x - close)))
        else:
            nearest_support.append(None)
        
        if resistance_prices:
            nearest_resistance.append(min(resistance_prices, key=lambda x: abs(x - close)))
        else:
            nearest_resistance.append(None)

    df['nearest_support'] = nearest_support
    df['nearest_resistance'] = nearest_resistance
    df['dist_to_support'] = df['close'] - df['nearest_support']
    df['dist_to_resistance'] = df['nearest_resistance'] - df['close']

    return df

def generate_swing_labels(df, target_pct=0.05, window=10):
    """
    Adds a column 'target_hit' = 1 if price hits +target_pct within window days, else 0.
    Keeps the same row length as df with no NaNs.
    """
    df = df.copy()
    target_hit = []

    for i in range(len(df)):
        if i + window >= len(df):
            # Not enough future data to check
            target_hit.append(None)
        else:
            entry_price = df['close'].iloc[i]
            future_high = df['high'].iloc[i+1:i+1+window].max()
            if (future_high - entry_price) / entry_price >= target_pct:
                target_hit.append(1)
            else:
                target_hit.append(0)

    df['target_hit'] = target_hit
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
