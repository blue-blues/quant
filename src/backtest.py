import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from config import config
from model import Agent
from data_loader import DataLoader
from environment import TradingEnvironment
import torch
class Backtest:
    def __init__(self, model_path='best_model.pth', initial_balance=10000):
        """Initialize backtest with model and configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.cfg = config()
        self.data_loader = DataLoader(self.cfg)
        self.env = TradingEnvironment(self.data_loader, initial_balance)
        
        # Setup GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        
        self.logger.info(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        # Initialize components with device
        self.agent = Agent(self.cfg, self.device)
        
        try:
            self.agent.load(model_path)
            self.logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            self.logger.warning(f"Could not load model from {model_path}: {str(e)}")
            self.logger.info("Using new model with random weights")
        
        # Initialize portfolio tracking
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position_size = 0
        self.last_buy_price = 0  # Track last buy price for stop loss
        
        # Fee settings
        self.maker_fee = self.cfg.MAKER_FEE
        self.taker_fee = self.cfg.TAKER_FEE
        self.fixed_fee = self.cfg.FIXED_FEE
        self.slippage = self.cfg.SLIPPAGE
        
        # Performance tracking
        self.trades = []
        self.portfolio_values = []
        self.total_fees = 0.0
        self.total_slippage = 0.0
        
        # Add device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
    def calculate_trade_costs(self, trade_type: str, price: float, quantity: float) -> tuple:
        """Calculate trading fees and slippage costs"""
        trade_value = price * quantity
        
        # Calculate fee based on trade type
        fee_rate = self.taker_fee if trade_type == 'market' else self.maker_fee
        fee = trade_value * fee_rate + self.fixed_fee
        
        # Calculate slippage
        slippage_cost = trade_value * self.slippage
        
        return fee, slippage_cost
        
    def execute_trade(self, action: int, price: float, timestamp) -> tuple:
        """Execute trade with costs and update portfolio"""
        if action == 0:  # HOLD
            return 0, 0
            
        trade_size = 0
        trade_fee = 0
        slippage_cost = 0
        
        if action == 1:  # BUY
            if self.position_size == 0:  # Only buy if no position
                trade_size = self.balance / price
                trade_fee, slippage_cost = self.calculate_trade_costs('market', price, trade_size)
                total_cost = (trade_size * price) + trade_fee + slippage_cost
                
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.position_size = trade_size
                    self.total_fees += trade_fee
                    self.total_slippage += slippage_cost
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': price,
                        'quantity': trade_size,
                        'fee': trade_fee,
                        'slippage': slippage_cost,
                        'cost': total_cost,
                        'balance': self.balance,
                        'portfolio_value': self.balance + (self.position_size * price)
                    })
                    
        elif action == 2:  # SELL
            if self.position_size > 0:  # Only sell if position exists
                trade_size = self.position_size
                trade_fee, slippage_cost = self.calculate_trade_costs('market', price, trade_size)
                
                gross_proceeds = trade_size * price
                net_proceeds = gross_proceeds - trade_fee - slippage_cost
                
                self.balance += net_proceeds
                self.position_size = 0
                self.total_fees += trade_fee
                self.total_slippage += slippage_cost
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': price,
                    'quantity': trade_size,
                    'fee': trade_fee,
                    'slippage': slippage_cost,
                    'proceeds': net_proceeds,
                    'balance': self.balance,
                    'portfolio_value': self.balance
                })
                
        return trade_fee, slippage_cost
    
    def run(self):
        """Run backtest simulation"""
        state = self.env.reset()
        done = False
        
        # Track trades and portfolio
        trades = []
        portfolio_values = []
        
        # Much more aggressive trading thresholds
        buy_threshold = 0.2   # Even lower threshold to encourage buying
        sell_threshold = 0.2  # Even lower threshold to encourage selling
        min_holding_period = 4  # Minimum hours to hold a position
        holding_time = 0
        
        while not done:
            current_step = self.env.current_step
            current_price = float(self.env.data.iloc[current_step]['close'])
            current_time = self.env.data.index[current_step]
            
            # Calculate price movement signals
            price_change = 0
            if current_step > 0:
                prev_price = float(self.env.data.iloc[current_step - 1]['close'])
                price_change = (current_price - prev_price) / prev_price
            
            # Get model's action probabilities
            if self.device.type == 'cuda':
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    # Use updated autocast syntax
                    with torch.amp.autocast(device_type='cuda'):
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = self.agent.model(state_tensor)
                        action_probs = torch.softmax(q_values, dim=1).squeeze().cpu()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.agent.model(state_tensor)
                    action_probs = torch.softmax(q_values, dim=1).squeeze()
            
            # Determine action based on position and market conditions
            if self.position_size == 0:  # No position
                if action_probs[1] > buy_threshold and price_change > 0:
                    action = 1  # BUY
                    holding_time = 0
                else:
                    action = 0  # HOLD
            else:  # Have position
                holding_time += 1
                # Only allow selling after minimum holding period
                if holding_time >= min_holding_period:
                    if action_probs[2] > sell_threshold or price_change < -0.01:  # 1% stop loss
                        action = 2  # SELL
                    else:
                        action = 0  # HOLD
                else:
                    action = 0  # HOLD
            
            # Execute trade
            if action != 0:  # If not holding
                trade_fee, slippage = self.execute_trade(action, current_price, current_time)
                if action == 1:
                    self.last_buy_price = current_price
            
            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(current_price)
            portfolio_values.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'price': current_price,
                'balance': self.balance,
                'position_size': self.position_size,
                'action': action
            })
            
            # Get next state
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            
            # Force close position at end of backtest
            if done and self.position_size > 0:
                self.execute_trade(2, current_price, current_time)
        
        return pd.DataFrame(self.trades), pd.DataFrame(portfolio_values)

    def calculate_portfolio_value(self, current_price):
        """Calculate current portfolio value including cash and positions"""
        position_value = self.position_size * current_price
        return self.balance + position_value
        
    def analyze_results(self, trades_df, portfolio_df):
        """Calculate and return backtest metrics"""
        try:
            metrics = {}
            
            # Basic metrics
            initial_value = self.initial_balance
            final_value = portfolio_df['portfolio_value'].iloc[-1]
            
            # Calculate cumulative returns
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
            
            # Calculate drawdown
            portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
            
            # Core metrics
            metrics['Initial Value'] = f"${initial_value:.2f}"
            metrics['Final Value'] = f"${final_value:.2f}"
            metrics['Total Return'] = f"{((final_value / initial_value) - 1):.2%}"
            metrics['Number of Trades'] = len(trades_df)
            
            if len(trades_df) > 0:
                # Trading metrics
                metrics['Total Fees'] = f"${self.total_fees:.2f}"
                metrics['Total Slippage'] = f"${self.total_slippage:.2f}"
                metrics['Average Fee/Trade'] = f"${self.total_fees/len(trades_df):.2f}"
                
                # Risk metrics
                metrics['Max Drawdown'] = f"{portfolio_df['drawdown'].min():.2%}"
                
                # Return metrics
                returns = portfolio_df['returns'].dropna()
                if len(returns) > 0:
                    metrics['Annualized Return'] = f"{returns.mean() * 252 * 100:.2f}%"
                    metrics['Annualized Volatility'] = f"{returns.std() * np.sqrt(252) * 100:.2f}%"
                    metrics['Sharpe Ratio'] = f"{(returns.mean() / returns.std()) * np.sqrt(252):.2f}"
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def plot_results(self, trades_df, portfolio_df):
        """Plot backtest results"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot portfolio value
            plt.subplot(2, 1, 1)
            plt.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], label='Portfolio Value', color='blue')
            
            if len(trades_df) > 0:
                # Add buy/sell markers
                buys = trades_df[trades_df['action'] == 'BUY']
                sells = trades_df[trades_df['action'] == 'SELL']
                
                plt.scatter(buys['timestamp'], buys['portfolio_value'], 
                           marker='^', color='green', label='Buy')
                plt.scatter(sells['timestamp'], sells['portfolio_value'], 
                           marker='v', color='red', label='Sell')
            
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            
            # Calculate and plot drawdown if not already calculated
            if 'drawdown' not in portfolio_df.columns:
                portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
                portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
            
            # Plot drawdown
            plt.subplot(2, 1, 2)
            plt.plot(portfolio_df['timestamp'], portfolio_df['drawdown'] * 100, color='red')
            plt.fill_between(portfolio_df['timestamp'], portfolio_df['drawdown'] * 100, 0, 
                            color='red', alpha=0.3)
            plt.title('Portfolio Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")
            plt.close()

def main():
    # Initialize and run backtest
    try:
        backtest = Backtest(initial_balance=10000)
        trades_df, portfolio_df = backtest.run()
        
        # Analyze results
        metrics = backtest.analyze_results(trades_df, portfolio_df)
        
        # Print metrics
        print("\nBacktest Results:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
        # Plot results
        backtest.plot_results(trades_df, portfolio_df)
        
        # Save detailed results
        trades_df.to_csv('trades.csv')
        portfolio_df.to_csv('portfolio.csv')
        
    except Exception as e:
        logging.error(f"Backtest failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
