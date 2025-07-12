# unified_data_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UnifiedDataProcessor:
    """
    Unified data processing pipeline that ensures consistency 
    between training and prediction phases
    """
    
    def __init__(self, config=None):
        """Initialize with configuration parameters"""
        self.config = config or {
            'sr_window': 10,
            'target_pct': 0.05,
            'stop_loss_pct': 0.03,
            'prediction_window': 10,
            'min_price': 25,
            'min_volume': 100000,
            'rsi_period': 14,
            'bb_period': 20,
            'ema_periods': [20, 50],
            'volume_periods': [10, 20, 50],
            'volatility_periods': [5, 10, 20]
        }
        
        self.feature_columns = None
        self.scaler = None
        self.is_fitted = False
        
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators consistently"""
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands Position
        sma = df['close'].rolling(self.config['bb_period']).mean()
        std = df['close'].rolling(self.config['bb_period']).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        df['bb_position'] = ((df['close'] - lower_band) / (upper_band - lower_band)).clip(0, 1)
        
        return df
    
    def calculate_candle_features(self, df):
        """Calculate candle pattern features"""
        df = df.copy()
        
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Normalized features
        df['body_to_range'] = df['body'] / (df['range'] + 1e-8)
        df['upper_shadow_to_range'] = df['upper_shadow'] / (df['range'] + 1e-8)
        df['lower_shadow_to_range'] = df['lower_shadow'] / (df['range'] + 1e-8)
        
        # Pattern detection
        df['is_doji'] = (df['body_to_range'] < 0.1).astype(int)
        df['is_hammer'] = ((df['body_to_range'] < 0.3) & 
                          (df['lower_shadow_to_range'] > 0.6)).astype(int)
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        return df
    
    def calculate_volume_features(self, df):
        """Calculate comprehensive volume features"""
        df = df.copy()
        
        # Volume moving averages
        for period in self.config['volume_periods']:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_ma_{period}'] + 1)
        
        # Volume spikes
        df['volume_spike'] = (df['volume_ratio_20'] > 2.0).astype(int)
        df['high_volume_spike'] = (df['volume_ratio_20'] > 3.0).astype(int)
        
        # Volume trends
        df['volume_trend_5'] = df['volume'].rolling(5).apply(
            lambda x: 1 if len(x) >= 5 and x.iloc[-1] > x.iloc[0] else 0
        )
        df['volume_trend_10'] = df['volume'].rolling(10).apply(
            lambda x: 1 if len(x) >= 10 and x.iloc[-1] > x.iloc[0] else 0
        )
        
        # VWAP calculations
        for period in [5, 10, 20]:
            df[f'vwap_{period}'] = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
            df[f'price_vs_vwap_{period}'] = (df['close'] - df[f'vwap_{period}']) / df['close']
        
        # Volume breakout
        df['volume_breakout'] = (
            (df['volume_ratio_20'] > 1.5) & 
            (df['close'] > df['high'].rolling(10).max().shift(1))
        ).astype(int)
        
        # OBV
        df['obv'] = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, 
                     np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
        df['obv_ma_10'] = df['obv'].rolling(10).mean()
        df['obv_divergence'] = (df['obv'] - df['obv_ma_10']) / (df['obv_ma_10'] + 1)
        
        # Volume-confirmed trend
        df['volume_confirmed_trend'] = (
            (df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]) > 0) & 
            (df['volume_ratio_20'] > 1.2)
        ).astype(int)
        
        return df
    
    def calculate_price_features(self, df):
        """Calculate price-based features"""
        df = df.copy()
        
        # Basic price features
        df['daily_return'] = df['close'].pct_change()
        df['range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Multiple timeframe volatility
        for period in self.config['volatility_periods']:
            df[f'volatility_{period}'] = df['daily_return'].rolling(period).std()
            df[f'return_mean_{period}'] = df['daily_return'].rolling(period).mean()
        
        # EMA relationships
        df['price_above_ema20'] = (df['close'] > df['ema20']).astype(int)
        df['price_above_ema50'] = (df['close'] > df['ema50']).astype(int)
        df['ema20_above_ema50'] = (df['ema20'] > df['ema50']).astype(int)
        
        # EMA distances
        df['ema20_distance'] = (df['close'] - df['ema20']) / df['close']
        df['ema50_distance'] = (df['close'] - df['ema50']) / df['close']
        df['ema_spread'] = (df['ema20'] - df['ema50']) / df['close']
        
        # Trend strength
        df['trend_strength'] = df['close'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 10 else 0
        )
        
        return df
    
    def calculate_support_resistance(self, df):
        """Calculate support and resistance levels consistently"""
        df = df.copy()
        window = self.config['sr_window']
        
        # Rolling max/min
        rolling_max = df['high'].rolling(window=2*window+1, center=True).max()
        rolling_min = df['low'].rolling(window=2*window+1, center=True).min()
        
        swing_highs = df['high'] == rolling_max
        swing_lows = df['low'] == rolling_min
        
        # Get support and resistance levels
        support_levels = []
        resistance_levels = []
        
        for idx in df[swing_lows].index:
            level_price = df.loc[idx, 'low']
            strength = sum(abs(df['low'] - level_price) / level_price < 0.01)
            support_levels.append((level_price, strength))
        
        for idx in df[swing_highs].index:
            level_price = df.loc[idx, 'high']
            strength = sum(abs(df['high'] - level_price) / level_price < 0.01)
            resistance_levels.append((level_price, strength))
        
        # Add nearest S/R features
        nearest_support = []
        nearest_resistance = []
        support_strength = []
        resistance_strength = []
        
        for close in df['close']:
            if support_levels:
                nearest_sup = min(support_levels, key=lambda x: abs(x[0] - close))
                nearest_support.append(nearest_sup[0])
                support_strength.append(nearest_sup[1])
            else:
                nearest_support.append(close * 0.95)
                support_strength.append(1)
            
            if resistance_levels:
                nearest_res = min(resistance_levels, key=lambda x: abs(x[0] - close))
                nearest_resistance.append(nearest_res[0])
                resistance_strength.append(nearest_res[1])
            else:
                nearest_resistance.append(close * 1.05)
                resistance_strength.append(1)
        
        df['nearest_support'] = nearest_support
        df['nearest_resistance'] = nearest_resistance
        df['support_strength'] = support_strength
        df['resistance_strength'] = resistance_strength
        df['dist_to_support'] = df['close'] - df['nearest_support']
        df['dist_to_resistance'] = df['nearest_resistance'] - df['close']
        df['norm_dist_to_support'] = df['dist_to_support'] / df['close']
        df['norm_dist_to_resistance'] = df['dist_to_resistance'] / df['close']
        
        return df
    
    def generate_labels(self, df):
        """Generate swing trading labels consistently"""
        df = df.copy()
        target_pct = self.config['target_pct']
        window = self.config['prediction_window']
        stop_loss_pct = self.config['stop_loss_pct']
        
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
                
                if max_ret >= target_pct and min_ret > -stop_loss_pct:
                    target_hit.append(1)
                else:
                    target_hit.append(0)
        
        df['target_hit'] = target_hit
        df['max_return'] = max_return
        df['min_return'] = min_return
        
        return df
    
    def apply_quality_filters(self, df):
        """Apply quality stock filters"""
        if len(df) < 50:
            return False
            
        # Current price and volume
        current_price = df['close'].iloc[-1]
        avg_volume = df['volume'].rolling(50).mean().iloc[-1]
        
        # Quality checks
        price_ok = current_price >= self.config['min_price']
        volume_ok = avg_volume >= self.config['min_volume']
        
        # Volatility check
        price_volatility = df['close'].iloc[-20:].std() / df['close'].iloc[-20:].mean()
        volatility_ok = price_volatility < 0.5
        
        # Volume consistency
        volume_consistency = df['volume'].iloc[-20:].min() > 0
        
        return price_ok and volume_ok and volatility_ok and volume_consistency
    
    def process_single_symbol(self, df, include_labels=True):
        """Process a single symbol through the complete pipeline"""
        df = df.copy()
        
        # Apply quality filters first
        if not self.apply_quality_filters(df):
            return None
        
        # Step 1: Calculate all technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Step 2: Calculate candle features
        df = self.calculate_candle_features(df)
        
        # Step 3: Calculate volume features
        df = self.calculate_volume_features(df)
        
        # Step 4: Calculate price features
        df = self.calculate_price_features(df)
        
        # Step 5: Calculate support/resistance
        df = self.calculate_support_resistance(df)
        
        # Step 6: Generate labels (only for training)
        if include_labels:
            df = self.generate_labels(df)
            df = df.dropna(subset=['target_hit'])
        
        # Step 7: Handle missing values
        df = df.fillna(method='ffill')
        df = df.fillna(0)
        
        return df
    
    def fit_transform(self, df_all):
        """Fit the processor and transform training data"""
        print("🔧 Fitting unified data processor...")
        
        processed_data = []
        
        for symbol in df_all['symbol'].unique():
            df_symbol = df_all[df_all['symbol'] == symbol].copy()
            processed_df = self.process_single_symbol(df_symbol, include_labels=True)
            
            if processed_df is not None and len(processed_df) > 0:
                processed_data.append(processed_df)
        
        if not processed_data:
            raise ValueError("No valid data after processing")
        
        # Combine all processed data
        df_combined = pd.concat(processed_data, ignore_index=True)
        
        # Define feature columns (exclude target and metadata)
        exclude_cols = ['target_hit', 'max_return', 'min_return', 'timestamp', 'symbol']
        self.feature_columns = [col for col in df_combined.columns if col not in exclude_cols]
        
        # Fit scaler
        self.scaler = StandardScaler()
        X = df_combined[self.feature_columns]
        self.scaler.fit(X)
        
        self.is_fitted = True
        
        # Save processor state
        self.save_processor()
        
        print(f"✅ Processor fitted with {len(self.feature_columns)} features")
        return df_combined
    
    def transform(self, df_all, include_labels=False):
        """Transform data using fitted processor"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        processed_data = []
        
        for symbol in df_all['symbol'].unique():
            df_symbol = df_all[df_all['symbol'] == symbol].copy()
            processed_df = self.process_single_symbol(df_symbol, include_labels=include_labels)
            
            if processed_df is not None and len(processed_df) > 0:
                processed_data.append(processed_df)
        
        if not processed_data:
            return pd.DataFrame()
        
        df_combined = pd.concat(processed_data, ignore_index=True)
        
        # Ensure we have all required features
        missing_features = [col for col in self.feature_columns if col not in df_combined.columns]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            # Add missing features with default values
            for col in missing_features:
                df_combined[col] = 0
        
        return df_combined
    
    def get_scaled_features(self, df):
        """Get scaled features for model input"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before getting scaled features")
        
        X = df[self.feature_columns]
        return self.scaler.transform(X)
    
    def save_processor(self):
        """Save processor state"""
        processor_state = {
            'config': self.config,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(processor_state, 'unified_processor_state.pkl')
        joblib.dump(self.scaler, 'unified_scaler.pkl')
        
        print("✅ Processor state saved")
    
    def load_processor(self):
        """Load processor state"""
        try:
            processor_state = joblib.load('unified_processor_state.pkl')
            self.config = processor_state['config']
            self.feature_columns = processor_state['feature_columns']
            self.is_fitted = processor_state['is_fitted']
            
            self.scaler = joblib.load('unified_scaler.pkl')
            
            print("✅ Processor state loaded")
            return True
        except Exception as e:
            print(f"❌ Error loading processor: {e}")
            return False