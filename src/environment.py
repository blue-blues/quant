import gym
import numpy as np
import logging
from gym import spaces
from data_loader import DataLoader

class TradingEnvironment(gym.Env):
    def __init__(self, data_loader: DataLoader, initial_balance=10000):
        super().__init__()
        self.data_loader = data_loader
        self.initial_balance = initial_balance
        self.data = None
        self.current_step = None
        self.logger = logging.getLogger(__name__)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32  # Changed from 11 to 12
        )
    
    def reset(self):
        try:
            self.data = self.data_loader.fetch_historical_data()
            
            # Validate data columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in self.data.columns for col in required_columns):
                self.logger.warning("Missing required columns in data, using synthetic data")
                self.data = self.data_loader._generate_synthetic_data()
            
            # Validate data length
            if len(self.data) < self.data_loader.config.LOOKBACK_PERIOD:
                self.logger.warning("Insufficient data points, using synthetic data")
                self.data = self.data_loader._generate_synthetic_data()
            
            self.current_step = self.data_loader.config.LOOKBACK_PERIOD
            self.balance = self.initial_balance
            self.position = 0
            
            return self._get_observation()
            
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")
            self.data = self.data_loader._generate_synthetic_data()
            self.current_step = self.data_loader.config.LOOKBACK_PERIOD
            self.balance = self.initial_balance
            self.position = 0
            return self._get_observation()

    def step(self, action):
        try:
            if self.current_step >= len(self.data) - 1:
                return self._get_observation(), 0, True, {}
            
            current_price = float(self.data.iloc[self.current_step]['close'])
            next_price = float(self.data.iloc[self.current_step + 1]['close'])
            
            # Calculate immediate price change
            price_change = (next_price - current_price) / current_price
            
            # Initialize reward
            reward = 0
            
            if action == 1:  # BUY
                if self.position == 0:
                    self.position = self.balance / current_price
                    self.balance = 0
                    reward = 0  # Neutral reward for opening position
            
            elif action == 2:  # SELL
                if self.position > 0:
                    self.balance = self.position * current_price
                    profit = self.balance - self.initial_balance
                    reward = profit / self.initial_balance * 100  # Reward as percentage return
                    self.position = 0
            
            else:  # HOLD
                if self.position > 0:
                    # Reward based on price movement while holding
                    unrealized_profit = (self.position * next_price) - (self.position * current_price)
                    reward = (unrealized_profit / (self.position * current_price)) * 100
                else:
                    # Small negative reward for staying out of market during uptrend
                    if price_change > 0:
                        reward = -0.1
            
            self.current_step += 1
            done = self.current_step >= len(self.data) - 1
            
            return self._get_observation(), reward, done, {}
            
        except Exception as e:
            self.logger.error(f"Error in step: {str(e)}")
            return self._get_observation(), 0, True, {}

    def _get_observation(self):
        """Enhanced state observation with stronger signals"""
        try:
            window = self.data.iloc[self.current_step - self.data_loader.config.LOOKBACK_PERIOD:self.current_step]
            current_price = float(self.data.iloc[self.current_step]['close'])
            
            # Calculate price movements
            returns = window['close'].pct_change().fillna(0)
            volatility = returns.std()
            momentum = returns.mean()
            
            # Calculate trend signals
            short_ma = window['close'].rolling(7).mean().iloc[-1]
            long_ma = window['close'].rolling(21).mean().iloc[-1]
            trend = (current_price - long_ma) / long_ma
            
            features = [
                momentum * 100,  # Recent price momentum
                volatility * 100,  # Price volatility
                trend * 100,  # Trend strength
                (self.position * current_price / self.initial_balance) * 100,  # Position size %
                (self.balance / self.initial_balance) * 100,  # Cash ratio %
                
                window['rsi'].iloc[-1] / 100,  # RSI
                window['macd'].iloc[-1] / current_price * 100,  # MACD
                window['signal'].iloc[-1] / current_price * 100,  # Signal
                window['hist'].iloc[-1] / current_price * 100,  # Histogram
                
                (current_price / short_ma - 1) * 100,  # Short-term trend
                (current_price / long_ma - 1) * 100,  # Long-term trend
                (window['volume'].pct_change().mean()) * 100  # Volume trend
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error creating observation: {str(e)}")
            return np.zeros(12, dtype=np.float32)

    def _calculate_reward(self, action, price):
        if action == 0:  # HOLD
            reward = 0
        elif action == 1:  # BUY
            if self.position == 0:
                self.position = self.balance / price
                self.balance = 0
                reward = 0
            else:
                reward = -1  # Penalty for invalid action
        else:  # SELL
            if self.position > 0:
                self.balance = self.position * price
                self.position = 0
                reward = (self.balance - self.initial_balance) / self.initial_balance
            else:
                reward = -1  # Penalty for invalid action
        
        return reward
