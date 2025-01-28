from dataclasses import dataclass

@dataclass
class config:
    # API Configuration
    API_KEY = "your_binance_api_key"
    API_SECRET = "your_binance_api_secret"
    
    # Trading Parameters
    SYMBOL = "BTCUSDT"
    TIMEFRAME = "1h"
    LOOKBACK_PERIOD = 48      # Changed to 48 hours (2 days) for more reasonable window
    
    # Model Parameters
    LEARNING_RATE = 0.0005  # Reduced for stability
    BATCH_SIZE = 128  # Increased batch size
    GAMMA = 0.99
    MEMORY_SIZE = 100000  # Increased memory size
    
    # Features
    MA_PERIODS = [7, 14, 21]
    RSI_PERIOD = 14
    MACD_PARAMS = (12, 26, 9)
    
    # Enhanced Model Parameters
    INPUT_SIZE = 12  # Changed from 11 to 12 to match state dimensions
    HIDDEN_SIZES = [128, 256, 128]
    
    # Additional Technical Indicators
    VOLUME_MA_PERIODS = [5, 10, 20]
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    ATR_PERIOD = 14
    
    # Training Parameters
    TARGET_UPDATE_STEPS = 1000
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 10000
    
    # Validation Parameters
    VALIDATION_INTERVAL = 10
    EARLY_STOPPING_PATIENCE = 20

    # Data Loading Parameters
    USE_OFFLINE_DATA = True
    MAX_RETRIES = 3
    DATA_CACHE_DAYS = 1  # How often to refresh cached data
    
    # Data Download Parameters
    DATA_SOURCE = "binance_vision"  # Options: "binance_api", "binance_vision"
    DOWNLOAD_MONTHS = 12  # Number of months of data to download
    DATA_CACHE_DAYS = 1  # Download new data if cache is older than this
    
    # Data Download Settings
    DOWNLOAD_START_DATE = "2023-01-01"
    DOWNLOAD_END_DATE = None  # None means current date
    DATA_SOURCE = "binance_vision"  # "binance_api" or "binance_vision"
    
    # Data Cache Settings
    CACHE_DIRECTORY = "data"
    CACHE_EXPIRY_DAYS = 1
    
    # Futures Trading Parameters
    LEVERAGE = 1  # Initial leverage
    POSITION_SIZE = 0.1  # Position size as fraction of balance
    MAX_POSITIONS = 3  # Maximum number of concurrent positions
    USE_STOP_LOSS = True
    STOP_LOSS_PCT = 0.02  # 2% stop loss
    
    # Trading Fee Configuration
    MAKER_FEE = 0.001  # 0.1% maker fee
    TAKER_FEE = 0.001  # 0.1% taker fee
    FIXED_FEE = 0.0    # Fixed fee per trade
    SLIPPAGE = 0.0005  # 0.05% slippage assumption
    
    # Add GPU settings
    USE_GPU = True
    CUDA_DEVICE = 0  # Which GPU to use if multiple available
    
    # Add batch processing settings for GPU
    GPU_BATCH_SIZE = 512  # Larger batch size for GPU processing
    NUM_WORKERS = 4  # Number of data loading workers
    PIN_MEMORY = True  # Pin memory for faster data transfer to GPU
    
    # Additional Technical Indicators