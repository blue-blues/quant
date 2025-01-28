from config import config
from data_loader import DataLoader
from environment import TradingEnvironment
from model import Agent
import logging
from pathlib import Path
import torch
import gc

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def setup_gpu():
    """Configure GPU settings"""
    if torch.cuda.is_available():
        # Set memory growth
        torch.cuda.empty_cache()
        # Enable cudnn benchmark for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    return device

def load_or_download_data(cfg, logger):
    """Load existing data or download if necessary"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Try to use the most recent downloaded data first
    downloaded_files = list(data_dir.glob(f"{cfg.SYMBOL}_{cfg.TIMEFRAME}_*.csv"))
    if downloaded_files:
        latest_file = max(downloaded_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using existing data file: {latest_file}")
        return latest_file
    
    # If no data exists, force a new download
    logger.info("No existing data found, initiating download...")
    data_loader = DataLoader(cfg, use_offline_data=False)
    data = data_loader.fetch_historical_data()
    if data is not None:
        logger.info("Successfully downloaded new data")
        return data
    else:
        logger.warning("Download failed, using synthetic data")
        return None

def main():
    logger = setup_logging()
    cfg = config()
    
    # Setup GPU
    device = setup_gpu()
    
    # Try to load or download data
    data_file = load_or_download_data(cfg, logger)
    
    # Initialize components with the correct data source
    data_loader = DataLoader(cfg, use_offline_data=True)
    env = TradingEnvironment(data_loader)
    agent = Agent(cfg, device)
    
    best_reward = float('-inf')
    
    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        total_loss = 0
        
        while not done:
            epsilon = max(0.01, 0.1 - 0.01 * episode / 200)  # Decay epsilon
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.append((state, action, reward, next_state, done))
            loss = agent.train(cfg.BATCH_SIZE)
            
            if loss is not None:
                total_loss += loss
            
            state = next_state
            total_reward += reward
            steps += 1
        
        avg_loss = total_loss / steps if steps > 0 else 0
        logger.info(f"Episode {episode}, Steps: {steps}, Total Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('best_model.pth')
            logger.info(f"New best model saved with reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()
