# Quantitative Trading Bot

A deep reinforcement learning-based cryptocurrency trading bot using DQN (Deep Q-Network) architecture.

## Project Structure

```
quant/
├── src/
│   ├── config.py      - Configuration parameters
│   ├── data_loader.py - Data fetching and processing
│   ├── environment.py - Trading environment
│   ├── model.py       - DQN model and agent
│   └── main.py        - Main execution script
```

## Features

- Real-time cryptocurrency trading using Binance API
- Deep Q-Network (DQN) for trading decisions
- Technical indicators integration:
  - Moving Averages (7, 14, 21 periods)
  - RSI (14 periods)
  - MACD (12, 26, 9)
- Experience replay for stable training
- Configurable trading parameters

## Setup

1. Install requirements:
```bash
pip install torch pandas numpy gym binance-connector
```

2. Configure API keys:
   - Open `src/config.py`
   - Replace placeholder API credentials with your Binance API keys:
     ```python
     API_KEY = "your_binance_api_key"
     API_SECRET = "your_binance_api_secret"
     ```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd quant
```

2. Install required packages:
```bash
pip install torch pandas numpy gym binance-connector python-dotenv
```

## Configuration

1. Create a `.env` file in the project root:
```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

2. Adjust trading parameters in `src/config.py`:
```python
SYMBOL = "BTCUSDT"        # Trading pair
TIMEFRAME = "1h"          # Candlestick interval
LOOKBACK_PERIOD = 100     # Historical data window
BATCH_SIZE = 64          # Training batch size
LEARNING_RATE = 0.001    # Model learning rate
```

## Usage

Run the trading bot:
```bash
python src/main.py
```

## Model Usage

### Training Mode
```bash
python src/main.py --mode train --episodes 1000
```

### Testing Mode
```bash
python src/main.py --mode test --model_path best_model.pth
```

### Backtesting
```bash
python src/main.py --mode backtest --start_date 2023-01-01 --end_date 2023-12-31
```

## Components

### Data Loader
- Fetches historical data from Binance
- Calculates technical indicators
- Preprocesses data for the model

### Trading Environment
- Implements OpenAI Gym interface
- Manages trading state and actions
- Calculates trading rewards

### DQN Model
- 3-layer neural network
- Actions: Hold, Buy, Sell
- Uses experience replay for training

### Configuration
- Adjustable hyperparameters
- Trading pairs and timeframes
- Technical indicator settings

## Model Architecture

### Input Features (11 dimensions):
- Price metrics (4):
  - Close price change
  - Volume change
  - Account balance ratio
  - Position size
- Technical indicators (7):
  - Moving averages (3)
  - RSI (1)
  - MACD components (3)

### Actions:
- 0: HOLD - maintain current position
- 1: BUY - purchase asset using full balance
- 2: SELL - sell entire position

### Reward Structure:
- HOLD: 0
- BUY: 0 (successful) / -1 (invalid)
- SELL: ROI (successful) / -1 (invalid)

## Performance Monitoring

Track model performance using:
```bash
tensorboard --logdir runs/
```

Key metrics:
- Total reward per episode
- Average trade return
- Position holding time
- Win rate

## Model Checkpoints

Models are saved automatically when achieving new best rewards:
```python
best_model.pth          # Best performing model
latest_checkpoint.pth   # Latest training state
```

Load a saved model:
```python
from model import Agent
agent = Agent(config)
agent.load('best_model.pth')
```

## License

MIT License
