import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import time
import os
import json
from pathlib import Path
import requests
import zipfile
import io
from config import config
import logging

class DataLoader:
    def __init__(self, config: config, use_offline_data=True, max_retries=3):
        self.config = config
        self.max_retries = max_retries
        self.use_offline_data = use_offline_data
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Base URLs for different Binance endpoints
        self.spot_base_url = "https://data.binance.vision/data/spot/monthly/klines"
        self.futures_base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
        self.api_base_url = "https://fapi.binance.com"
        
        # API endpoints
        self.api_endpoints = {
            'klines': "/fapi/v1/klines",
            'ticker': "/fapi/v1/ticker/24hr",
            'depth': "/fapi/v1/depth",
            'trades': "/fapi/v1/trades",
            'agg_trades': "/fapi/v1/aggTrades",
            'open_interest': "/fapi/v1/openInterest",
            'funding_rate': "/fapi/v1/fundingRate",
            'top_long_short': "/futures/data/topLongShortAccountRatio",
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def fetch_all_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive market data including futures-specific metrics"""
        try:
            # Fetch different types of data
            klines_data = self._fetch_klines()
            market_data = self._fetch_market_metrics()
            orderbook_data = self._fetch_orderbook()
            
            # Combine all data
            combined_data = self._process_and_combine_data(
                klines_data, 
                market_data, 
                orderbook_data
            )
            
            # Save to cache
            self._save_offline_data(combined_data)
            
            return combined_data
            
        except Exception as e:
            print(f"Error fetching market data: {str(e)}")
            return self._load_offline_data()

    def _fetch_klines(self) -> pd.DataFrame:
        """Fetch detailed kline data"""
        endpoint = f"{self.api_base_url}{self.api_endpoints['klines']}"
        params = {
            'symbol': self.config.SYMBOL,
            'interval': self.config.TIMEFRAME,
            'limit': 1500  # Maximum allowed
        }
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')

    def _fetch_market_metrics(self) -> pd.DataFrame:
        """Fetch futures-specific metrics"""
        # Fetch funding rate
        funding_rate = self._fetch_endpoint('funding_rate', {
            'symbol': self.config.SYMBOL,
            'limit': 500
        })
        
        # Fetch open interest
        open_interest = self._fetch_endpoint('open_interest', {
            'symbol': self.config.SYMBOL
        })
        
        # Fetch long/short ratio
        ls_ratio = self._fetch_endpoint('top_long_short', {
            'symbol': self.config.SYMBOL,
            'period': '1h'
        })
        
        # Combine metrics
        metrics = pd.DataFrame({
            'funding_rate': funding_rate['fundingRate'],
            'open_interest': open_interest['openInterest'],
            'long_short_ratio': ls_ratio['longShortRatio']
        })
        
        return metrics

    def _fetch_orderbook(self, limit=1000) -> pd.DataFrame:
        """Fetch detailed orderbook data"""
        endpoint = f"{self.api_base_url}{self.api_endpoints['depth']}"
        params = {
            'symbol': self.config.SYMBOL,
            'limit': limit
        }
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        # Process orderbook
        bids_df = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
        asks_df = pd.DataFrame(data['asks'], columns=['price', 'quantity'])
        
        # Calculate orderbook metrics
        metrics = {
            'bid_ask_spread': float(asks_df['price'].iloc[0]) - float(bids_df['price'].iloc[0]),
            'bid_depth': bids_df['quantity'].astype(float).sum(),
            'ask_depth': asks_df['quantity'].astype(float).sum(),
            'bid_levels': len(bids_df),
            'ask_levels': len(asks_df)
        }
        
        return pd.Series(metrics)

    def _process_and_combine_data(self, klines: pd.DataFrame, 
                                market_metrics: pd.DataFrame,
                                orderbook: pd.Series) -> pd.DataFrame:
        """Combine and process all data sources"""
        df = klines.copy()
        
        # Add market metrics
        for col in market_metrics.columns:
            df[col] = market_metrics[col]
        
        # Add orderbook metrics
        for metric, value in orderbook.items():
            df[f'orderbook_{metric}'] = value
        
        # Calculate technical indicators
        df = self._add_technical_indicators(df)
        
        return df.dropna()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators"""
        # Existing indicators
        for ma_period in self.config.MA_PERIODS:
            df[f'ma_{ma_period}'] = df['close'].rolling(window=ma_period).mean()
        
        df['rsi'] = self._calculate_rsi(df['close'], self.config.RSI_PERIOD)
        df['macd'], df['signal'], df['hist'] = self._calculate_macd(df['close'])
        
        # New indicators
        df['volatility'] = df['close'].pct_change().rolling(window=24).std()
        df['volume_ma'] = df['volume'].rolling(window=24).mean()
        df['oi_change'] = df['open_interest'].pct_change()
        
        return df

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical data from Binance Vision or load from cache"""
        if self.use_offline_data:
            return self._load_offline_data()
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.DOWNLOAD_MONTHS * 30)
            
            monthly_data = []
            current_date = start_date
            
            while current_date <= end_date:
                df = self._download_monthly_klines(
                    self.config.SYMBOL,
                    self.config.TIMEFRAME,
                    current_date.year,
                    current_date.month
                )
                
                if df is not None:
                    monthly_data.append(df)
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            if not monthly_data:
                self.logger.warning("No historical data available, using synthetic data")
                return self._generate_synthetic_data()
            
            # Combine and process data
            combined_df = self._combine_monthly_data(monthly_data)
            if combined_df is None:
                self.logger.warning("Failed to process downloaded data, using synthetic data")
                return self._generate_synthetic_data()
            
            # Save to cache
            self._save_offline_data(combined_df)
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return self._generate_synthetic_data()

    def _download_monthly_klines(self, symbol: str, interval: str, year: int, month: int) -> pd.DataFrame:
        """Download monthly klines data from Binance Vision"""
        month_str = f"{month:02d}"
        symbol = symbol.upper()
        url = f"{self.spot_base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month_str}.zip"
        
        try:
            self.logger.info(f"Downloading from: {url}")
            response = requests.get(url)
            
            if response.status_code == 200:
                self.logger.info("Download successful")
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    csv_filename = zip_file.namelist()[0]
                    with zip_file.open(csv_filename) as csv_file:
                        df = pd.read_csv(
                            csv_file,
                            header=None,
                            names=[
                                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                'taker_buy_quote', 'ignored'
                            ]
                        )
                        
                        # Convert timestamp to datetime immediately
                        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Convert numeric columns
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        return df[['open', 'high', 'low', 'close', 'volume']]
            else:
                self.logger.warning(f"Download failed with status code: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error downloading from {url}: {e}")
            return None

    def _save_offline_data(self, df: pd.DataFrame):
        """Save data to CSV file with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{self.config.SYMBOL}_{self.config.TIMEFRAME}_{timestamp}.csv"
            filepath = self.data_dir / filename
            
            # Ensure index name is set before saving
            df.index.name = 'datetime'  # Changed from 'timestamp' to 'datetime'
            
            # Save with proper date format and index
            df.to_csv(filepath)
            
            # Save latest version
            latest_file = self.data_dir / f"{self.config.SYMBOL}_{self.config.TIMEFRAME}_latest.csv"
            df.to_csv(latest_file)
            
            self.logger.info(f"Saved data to {filepath} and {latest_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise

    def load_specific_file(self, filepath: Path) -> pd.DataFrame:
        """Load data from a specific file"""
        try:
            # Read CSV with explicit datetime index
            df = pd.read_csv(filepath)
            
            # Convert the datetime column to index
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Convert price columns to float
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.logger.info(f"Successfully loaded data from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading file {filepath}: {e}")
            return None

    def _combine_monthly_data(self, monthly_data: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine monthly data frames and ensure proper formatting"""
        if not monthly_data:
            return None
            
        try:
            # Combine all monthly data
            df = pd.concat(monthly_data)
            
            # Sort by datetime
            df = df.sort_index()
            
            # Add technical indicators
            df = self._process_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error combining monthly data: {e}")
            return None

    def _load_offline_data(self) -> pd.DataFrame:
        """Load most recent offline data or generate synthetic data"""
        # First try to load from downloaded test data
        test_files = list(self.data_dir.glob(f"test_{self.config.SYMBOL}_{self.config.TIMEFRAME}_*.csv"))
        if test_files:
            latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
            data = self.load_specific_file(latest_test_file)
            if data is not None:
                return data
        
        # Fall back to regular offline data
        latest_link = self.data_dir / f"{self.config.SYMBOL}_{self.config.TIMEFRAME}_latest.csv"
        if latest_link.exists():
            return self.load_specific_file(latest_link)
        
        # Generate synthetic data as last resort
        self.logger.warning("No offline data found, generating synthetic data")
        return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        periods = 1000
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='h')
        
        # Generate random walk prices
        np.random.seed(42)
        price = 100 * (1 + np.random.randn(periods).cumsum() * 0.02)
        volume = np.random.randint(100, 1000, periods)
        
        # Create synthetic data with realistic price movements
        df = pd.DataFrame({
            'open': price * (1 + np.random.randn(periods) * 0.001),
            'high': price * (1 + abs(np.random.randn(periods)) * 0.002),
            'low': price * (1 - abs(np.random.randn(periods)) * 0.002),
            'close': price,
            'volume': volume,
        }, index=dates)
        
        # Add technical indicators
        # Moving averages
        for ma_period in self.config.MA_PERIODS:
            df[f'ma_{ma_period}'] = df['close'].rolling(window=ma_period).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.config.RSI_PERIOD)
        
        # MACD
        df['macd'], df['signal'], df['hist'] = self._calculate_macd(df['close'])
        
        # Drop NaN values that result from indicator calculations
        return df.dropna()

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data from either real API or synthetic source"""
        try:
            df = df.copy()
            
            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort index
            df = df.sort_index()
            
            # Calculate technical indicators
            for ma_period in self.config.MA_PERIODS:
                df[f'ma_{ma_period}'] = df['close'].rolling(window=ma_period).mean()
            
            df['rsi'] = self._calculate_rsi(df['close'], self.config.RSI_PERIOD)
            df['macd'], df['signal'], df['hist'] = self._calculate_macd(df['close'])
            
            # Remove any NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return self._generate_synthetic_data()

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
