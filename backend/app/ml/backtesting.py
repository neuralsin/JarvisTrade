"""
Comprehensive Backtesting Framework
Walk-forward optimization, Monte Carlo simulation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Core backtesting engine with strategy execution
    """
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.0003,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
    
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_date = None
    
    def execute_signal(
        self,
        timestamp: datetime,
        symbol: str,
        signal: str,  # 'BUY', 'SELL', or 'HOLD'
        price: float,
        qty: int,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None
    ):
        """
        Execute trading signal
        """
        if signal == 'BUY' and symbol not in self.positions:
            # Apply slippage
            fill_price = price * (1 + self.slippage)
            cost = fill_price * qty
            commission_cost = cost * self.commission
            
            if cost + commission_cost <= self.capital:
                self.capital -= (cost + commission_cost)
                self.positions[symbol] = {
                    'entry_date': timestamp,
                    'entry_price': fill_price,
                    'qty': qty,
                    'stop_loss': stop_loss,
                    'target': target
                }
                
                logger.debug(f"[{timestamp}] BUY {symbol}: {qty} @ ₹{fill_price:.2f}")
        
        elif signal == 'SELL' and symbol in self.positions:
            position = self.positions[symbol]
            fill_price = price * (1 - self.slippage)
            proceeds = fill_price * position['qty']
            commission_cost = proceeds * self.commission
            
            self.capital += (proceeds - commission_cost)
            
            pnl = (fill_price - position['entry_price']) * position['qty'] - commission_cost
            pnl_pct = (fill_price / position['entry_price'] - 1) * 100
            
            self.trades.append({
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': timestamp,
                'entry_price': position['entry_price'],
                'exit_price': fill_price,
                'qty': position['qty'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration_days': (timestamp - position['entry_date']).days
            })
            
            del self.positions[symbol]
            
            logger.debug(f"[{timestamp}] SELL {symbol}: {position['qty']} @ ₹{fill_price:.2f}, P&L: ₹{pnl:.2f}")
    
    def check_stops_and_targets(self, timestamp: datetime, prices: Dict[str, float]):
        """
        Check if any positions hit stop loss or target
        """
        for symbol in list(self.positions.keys()):
            if symbol not in prices:
                continue
            
            position = self.positions[symbol]
            current_price = prices[symbol]
            
            # Check stop loss
            if position['stop_loss'] and current_price <= position['stop_loss']:
                self.execute_signal(timestamp, symbol, 'SELL', position['stop_loss'], position['qty'])
            
            # Check target
            elif position['target'] and current_price >= position['target']:
                self.execute_signal(timestamp, symbol, 'SELL', position['target'], position['qty'])
    
    def update_equity(self, timestamp: datetime, prices: Dict[str, float]):
        """
        Calculate current portfolio value
        """
        portfolio_value = self.capital
        
        # Add value of open positions
        for symbol, position in self.positions.items():
            if symbol in prices:
                portfolio_value += prices[symbol] * position['qty']
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': portfolio_value
        })
        
        return portfolio_value
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func
    ) -> Dict:
        """
        Run backtest with given strategy function
        
        Args:
            data: DataFrame with OHLCV and any indicators
            strategy_func: Function that returns (signal, stop, target) given a row
        
        Returns:
            dict: Backtest results
        """
        self.reset()
        
        for idx, row in data.iterrows():
            timestamp = row['ts_utc'] if 'ts_utc' in row else idx
            
            # Generate signal from strategy
            signal, stop, target = strategy_func(row)
            
            # Check existing positions
            current_prices = {row.get('symbol', 'UNKNOWN'): row['close']}
            self.check_stops_and_targets(timestamp, current_prices)
            
            # Execute new signal
            if signal in ['BUY', 'SELL']:
                qty = int(self.capital * 0.1 / row['close'])  # 10% position sizing
                self.execute_signal(
                    timestamp,
                    row.get('symbol', 'UNKNOWN'),
                    signal,
                    row['close'],
                    qty,
                    stop,
                    target
                )
            
            # Update equity curve
            self.update_equity(timestamp, current_prices)
        
        # Close remaining positions at final prices
        if len(data) > 0:
            final_row = data.iloc[-1]
            final_prices = {final_row.get('symbol', 'UNKNOWN'): final_row['close']}
            for symbol in list(self.positions.keys()):
                if symbol in final_prices:
                    self.execute_signal(
                        final_row['ts_utc'] if 'ts_utc' in final_row else data.index[-1],
                        symbol,
                        'SELL',
                        final_prices[symbol],
                        self.positions[symbol]['qty']
                    )
        
        return self.get_performance_metrics()
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'final_capital': self.capital
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio (assuming daily data)
        if len(equity_df) > 1:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_capital': self.capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor if profit_factor != float('inf') else 0,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration': trades_df['duration_days'].mean() if len(trades_df) > 0 else 0
        }


def walk_forward_optimization(
    data: pd.DataFrame,
    strategy_func,
    train_window: int = 252,  # 1 year
    test_window: int = 63,    # 3 months
    step_size: int = 21       # 1 month
) -> Dict:
    """
    Walk-forward optimization
    Train on rolling window, test on subsequent period
    """
    results = []
    
    for start_idx in range(0, len(data) - train_window - test_window, step_size):
        train_end_idx = start_idx + train_window
        test_end_idx = train_end_idx + test_window
        
        train_data = data.iloc[start_idx:train_end_idx]
        test_data = data.iloc[train_end_idx:test_end_idx]
        
        # Optimize on train data (simplified - just run backtest)
        train_engine = BacktestEngine()
        train_metrics = train_engine.run_backtest(train_data, strategy_func)
        
        # Test on out-of-sample data
        test_engine = BacktestEngine()
        test_metrics = test_engine.run_backtest(test_data, strategy_func)
        
        results.append({
            'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
            'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })
        
        logger.info(f"Walk-forward: Test return={test_metrics['total_return_pct']:.2f}%")
    
    # Aggregate results
    avg_test_return = np.mean([r['test_metrics']['total_return_pct'] for r in results])
    avg_test_sharpe = np.mean([r['test_metrics']['sharpe_ratio'] for r in results])
    
    return {
        'periods': results,
        'avg_test_return': avg_test_return,
        'avg_test_sharpe': avg_test_sharpe,
        'num_periods': len(results)
    }


def monte_carlo_simulation(
    trades: List[Dict],
    num_simulations: int = 10000,
    initial_capital: float = 100000.0
) -> Dict:
    """
    Monte Carlo simulation by randomly shuffling trade order
    """
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {'error': 'No trades to simulate'}
    
    simulated_final_capitals = []
    simulated_max_drawdowns = []
    
    for _ in range(num_simulations):
        # Randomly shuffle trades
        shuffled_trades = trades_df.sample(frac=1, replace=True)
        
        # Calculate equity curve
        capital = initial_capital
        equity_curve = [capital]
        
        for _, trade in shuffled_trades.iterrows():
            capital += trade['pnl']
            equity_curve.append(capital)
        
        simulated_final_capitals.append(capital)
        
        # Calculate drawdown
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak
        simulated_max_drawdowns.append(drawdown.min())
    
    # Calculate confidence intervals
    simulated_returns = [(c - initial_capital) / initial_capital for c in simulated_final_capitals]
    
    return {
        'num_simulations': num_simulations,
        'final_capital_p5': np.percentile(simulated_final_capitals, 5),
        'final_capital_p50': np.percentile(simulated_final_capitals, 50),
        'final_capital_p95': np.percentile(simulated_final_capitals, 95),
        'return_p5': np.percentile(simulated_returns, 5) * 100,
        'return_p50': np.percentile(simulated_returns, 50) * 100,
        'return_p95': np.percentile(simulated_returns, 95) * 100,
        'max_drawdown_p5': np.percentile(simulated_max_drawdowns, 5) * 100,
        'max_drawdown_p50': np.percentile(simulated_max_drawdowns, 50) * 100,
        'max_drawdown_p95': np.percentile(simulated_max_drawdowns, 95) * 100,
        'probability_of_profit': (np.array(simulated_returns) > 0).mean() * 100
    }
