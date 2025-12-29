"""
V2 Backtesting Framework - Non-Lying Backtest

CRITICAL V2 FIXES:
1. Entry at NEXT bar open (not same candle close)
2. Stops/targets evaluated using HIGH/LOW within each bar
3. Commission applied to both entry and exit
4. Slippage applied based on direction
5. Track signal bar vs entry bar separately
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class BacktestEngineV2:
    """
    V2 Backtesting Engine with realistic execution simulation.
    
    Key differences from V1:
    - Entry at NEXT bar open (simulates reality)
    - Intrabar stop/target evaluation using high/low
    - Direction-aware slippage
    - Pending orders queue
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
        self.pending_orders = []  # Orders waiting for next bar
        self.trades = []
        self.equity_curve = []
        self.current_bar_idx = 0
    
    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply slippage based on trade direction"""
        if direction == 'BUY':
            return price * (1 + self.slippage)
        else:  # SELL
            return price * (1 - self.slippage)
    
    def queue_order(
        self,
        signal_bar_idx: int,
        timestamp: datetime,
        symbol: str,
        direction: str,  # 'BUY' or 'SELL'
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        size_pct: float = 0.1
    ):
        """
        Queue an order for execution at NEXT bar open.
        
        This is the critical fix - we don't execute immediately.
        """
        self.pending_orders.append({
            'signal_bar_idx': signal_bar_idx,
            'signal_timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'stop_loss': stop_loss,
            'target': target,
            'size_pct': size_pct
        })
        logger.debug(f"[Bar {signal_bar_idx}] Queued {direction} {symbol} for next bar")
    
    def process_pending_orders(self, bar_idx: int, bar: pd.Series):
        """
        Execute pending orders at this bar's OPEN price.
        
        Args:
            bar_idx: Current bar index
            bar: Current bar data with open, high, low, close
        """
        symbol = bar.get('symbol', 'UNKNOWN')
        timestamp = bar.get('ts_utc', bar.name if hasattr(bar, 'name') else datetime.utcnow())
        
        orders_to_remove = []
        
        for i, order in enumerate(self.pending_orders):
            if order['symbol'] != symbol:
                continue
            
            # Check if already have position
            if order['direction'] == 'BUY' and symbol in self.positions:
                orders_to_remove.append(i)
                continue
            
            # Execute at this bar's OPEN with slippage
            open_price = bar['open']
            fill_price = self._apply_slippage(open_price, order['direction'])
            
            # Calculate position size
            qty = int(self.capital * order['size_pct'] / fill_price)
            if qty <= 0:
                orders_to_remove.append(i)
                continue
            
            if order['direction'] == 'BUY':
                cost = fill_price * qty
                commission_cost = cost * self.commission
                
                if cost + commission_cost <= self.capital:
                    self.capital -= (cost + commission_cost)
                    self.positions[symbol] = {
                        'signal_bar_idx': order['signal_bar_idx'],
                        'entry_bar_idx': bar_idx,
                        'entry_date': timestamp,
                        'entry_price': fill_price,
                        'qty': qty,
                        'stop_loss': order['stop_loss'],
                        'target': order['target'],
                        'direction': 'LONG'
                    }
                    logger.debug(f"[Bar {bar_idx}] ENTRY {symbol}: {qty} @ ₹{fill_price:.2f} "
                                f"(signal at bar {order['signal_bar_idx']})")
            
            orders_to_remove.append(i)
        
        # Remove processed orders (reverse order to maintain indices)
        for i in sorted(orders_to_remove, reverse=True):
            self.pending_orders.pop(i)
    
    def check_intrabar_exit(self, bar_idx: int, bar: pd.Series) -> Optional[Dict]:
        """
        Check if stop/target hit using bar's HIGH/LOW.
        
        This is the critical fix - we check intrabar, not just close.
        """
        symbol = bar.get('symbol', 'UNKNOWN')
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        high = bar['high']
        low = bar['low']
        timestamp = bar.get('ts_utc', bar.name)
        
        exit_price = None
        exit_reason = None
        
        if position['direction'] == 'LONG':
            # Check stop first (conservative)
            if position['stop_loss'] and low <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_reason = 'STOP_LOSS'
            elif position['target'] and high >= position['target']:
                exit_price = position['target']
                exit_reason = 'TARGET'
        else:  # SHORT
            if position['stop_loss'] and high >= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_reason = 'STOP_LOSS'
            elif position['target'] and low <= position['target']:
                exit_price = position['target']
                exit_reason = 'TARGET'
        
        if exit_price:
            return self._close_position(symbol, exit_price, timestamp, bar_idx, exit_reason)
        
        return None
    
    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: datetime,
        bar_idx: int,
        exit_reason: str
    ) -> Dict:
        """Close position and record trade"""
        position = self.positions[symbol]
        
        # Apply slippage on exit
        fill_price = self._apply_slippage(exit_price, 'SELL')
        proceeds = fill_price * position['qty']
        commission_cost = proceeds * self.commission
        
        self.capital += (proceeds - commission_cost)
        
        pnl = (fill_price - position['entry_price']) * position['qty'] - commission_cost * 2  # Entry + exit commission
        pnl_pct = (fill_price / position['entry_price'] - 1) * 100
        bars_held = bar_idx - position['entry_bar_idx']
        
        trade_record = {
            'symbol': symbol,
            'signal_bar_idx': position['signal_bar_idx'],
            'entry_bar_idx': position['entry_bar_idx'],
            'exit_bar_idx': bar_idx,
            'entry_date': position['entry_date'],
            'exit_date': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': fill_price,
            'qty': position['qty'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade_record)
        del self.positions[symbol]
        
        logger.debug(f"[Bar {bar_idx}] EXIT {symbol}: {position['qty']} @ ₹{fill_price:.2f}, "
                    f"P&L: ₹{pnl:.2f} ({exit_reason})")
        
        return trade_record
    
    def update_equity(self, timestamp: datetime, close_prices: Dict[str, float]):
        """Calculate current portfolio value"""
        portfolio_value = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in close_prices:
                portfolio_value += close_prices[symbol] * position['qty']
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': portfolio_value
        })
        
        return portfolio_value
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable
    ) -> Dict:
        """
        Run V2 backtest with next-bar entry and intrabar stops.
        
        Args:
            data: DataFrame with OHLCV (must have open, high, low, close, ts_utc)
            strategy_func: Function(row) -> (signal, stop, target) where signal is 'BUY'/'SELL'/'HOLD'
        
        Returns:
            dict: Performance metrics
        """
        self.reset()
        
        if len(data) < 2:
            logger.warning("Insufficient data for backtest")
            return self.get_performance_metrics()
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        data = data.reset_index(drop=True)
        
        for idx in range(len(data)):
            row = data.iloc[idx]
            timestamp = row.get('ts_utc', idx)
            symbol = row.get('symbol', 'UNKNOWN')
            
            # Step 1: Process any pending orders at this bar's OPEN
            if idx > 0:  # Can't execute on first bar
                self.process_pending_orders(idx, row)
            
            # Step 2: Check intrabar stop/target on existing positions
            self.check_intrabar_exit(idx, row)
            
            # Step 3: Generate signal from strategy
            signal, stop, target = strategy_func(row)
            
            # Step 4: Queue new orders (will execute at NEXT bar)
            if signal == 'BUY' and symbol not in self.positions:
                self.queue_order(idx, timestamp, symbol, 'BUY', stop, target)
            
            # Step 5: Update equity
            self.update_equity(timestamp, {symbol: row['close']})
            
            self.current_bar_idx = idx
        
        # Close remaining positions at final close
        if len(data) > 0:
            final_row = data.iloc[-1]
            final_timestamp = final_row.get('ts_utc', len(data) - 1)
            for symbol in list(self.positions.keys()):
                self._close_position(
                    symbol,
                    final_row['close'],
                    final_timestamp,
                    len(data) - 1,
                    'END_OF_BACKTEST'
                )
        
        return self.get_performance_metrics()
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'final_capital': self.capital,
                'is_honest': True  # V2 flag
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
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio
        if len(equity_df) > 1:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # V2-specific: Average bars held, exit reason distribution
        avg_bars_held = trades_df['bars_held'].mean() if 'bars_held' in trades_df else 0
        
        exit_reasons = {}
        if 'exit_reason' in trades_df:
            exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
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
            'avg_bars_held': avg_bars_held,
            'exit_reasons': exit_reasons,
            'is_honest': True  # V2 flag - this backtest doesn't lie
        }


# =============================================================================
# LEGACY SUPPORT - V1 BacktestEngine kept for backward compatibility
# =============================================================================

class BacktestEngine:
    """
    DEPRECATED: V1 Backtesting Engine
    Use BacktestEngineV2 for accurate results.
    
    This class is kept for backward compatibility but logs warnings.
    """
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.0003,
        slippage: float = 0.0005
    ):
        logger.warning("⚠️ DEPRECATED: Using V1 BacktestEngine which has known issues. "
                      "Use BacktestEngineV2 for accurate results.")
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
        signal: str,
        price: float,
        qty: int,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None
    ):
        """Execute trading signal - V1 (same-bar entry)"""
        if signal == 'BUY' and symbol not in self.positions:
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
                'duration_days': (timestamp - position['entry_date']).days if isinstance(timestamp, datetime) else 0
            })
            
            del self.positions[symbol]
    
    def check_stops_and_targets(self, timestamp: datetime, prices: Dict[str, float]):
        """Check if any positions hit stop loss or target"""
        for symbol in list(self.positions.keys()):
            if symbol not in prices:
                continue
            
            position = self.positions[symbol]
            current_price = prices[symbol]
            
            if position['stop_loss'] and current_price <= position['stop_loss']:
                self.execute_signal(timestamp, symbol, 'SELL', position['stop_loss'], position['qty'])
            elif position['target'] and current_price >= position['target']:
                self.execute_signal(timestamp, symbol, 'SELL', position['target'], position['qty'])
    
    def update_equity(self, timestamp: datetime, prices: Dict[str, float]):
        """Calculate current portfolio value"""
        portfolio_value = self.capital
        for symbol, position in self.positions.items():
            if symbol in prices:
                portfolio_value += prices[symbol] * position['qty']
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': portfolio_value
        })
        return portfolio_value
    
    def run_backtest(self, data: pd.DataFrame, strategy_func) -> Dict:
        """Run backtest - V1 (has known issues)"""
        self.reset()
        
        for idx, row in data.iterrows():
            timestamp = row['ts_utc'] if 'ts_utc' in row else idx
            signal, stop, target = strategy_func(row)
            
            current_prices = {row.get('symbol', 'UNKNOWN'): row['close']}
            self.check_stops_and_targets(timestamp, current_prices)
            
            if signal in ['BUY', 'SELL']:
                qty = int(self.capital * 0.1 / row['close'])
                self.execute_signal(timestamp, row.get('symbol', 'UNKNOWN'), signal,
                                   row['close'], qty, stop, target)
            
            self.update_equity(timestamp, current_prices)
        
        if len(data) > 0:
            final_row = data.iloc[-1]
            final_prices = {final_row.get('symbol', 'UNKNOWN'): final_row['close']}
            for symbol in list(self.positions.keys()):
                if symbol in final_prices:
                    self.execute_signal(
                        final_row['ts_utc'] if 'ts_utc' in final_row else data.index[-1],
                        symbol, 'SELL', final_prices[symbol], self.positions[symbol]['qty']
                    )
        
        return self.get_performance_metrics()
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {'total_trades': 0, 'total_return': 0, 'final_capital': self.capital, 'is_honest': False}
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
        
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
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
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration': trades_df['duration_days'].mean() if len(trades_df) > 0 else 0,
            'is_honest': False  # V1 flag - this backtest may lie
        }


# =============================================================================
# Walk-forward and Monte Carlo (unchanged, work with both engines)
# =============================================================================

def walk_forward_optimization(
    data: pd.DataFrame,
    strategy_func,
    train_window: int = 252,
    test_window: int = 63,
    step_size: int = 21,
    use_v2: bool = True
) -> Dict:
    """Walk-forward optimization with V2 engine by default"""
    results = []
    
    for start_idx in range(0, len(data) - train_window - test_window, step_size):
        train_end_idx = start_idx + train_window
        test_end_idx = train_end_idx + test_window
        
        train_data = data.iloc[start_idx:train_end_idx]
        test_data = data.iloc[train_end_idx:test_end_idx]
        
        EngineClass = BacktestEngineV2 if use_v2 else BacktestEngine
        
        train_engine = EngineClass()
        train_metrics = train_engine.run_backtest(train_data, strategy_func)
        
        test_engine = EngineClass()
        test_metrics = test_engine.run_backtest(test_data, strategy_func)
        
        results.append({
            'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
            'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })
    
    if not results:
        return {'periods': [], 'avg_test_return': 0, 'avg_test_sharpe': 0, 'num_periods': 0}
    
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
    """Monte Carlo simulation by randomly shuffling trade order"""
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {'error': 'No trades to simulate'}
    
    simulated_final_capitals = []
    simulated_max_drawdowns = []
    
    for _ in range(num_simulations):
        shuffled_trades = trades_df.sample(frac=1, replace=True)
        
        capital = initial_capital
        equity_curve = [capital]
        
        for _, trade in shuffled_trades.iterrows():
            capital += trade['pnl']
            equity_curve.append(capital)
        
        simulated_final_capitals.append(capital)
        
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / np.where(peak != 0, peak, 1)
        simulated_max_drawdowns.append(drawdown.min())
    
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
