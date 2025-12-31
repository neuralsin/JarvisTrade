"""
Unified Execution Simulator - V3 Critical Fix

SINGLE SOURCE OF TRUTH for trade execution simulation.
Used by: labeler, backtester, paper trader

If code paths diverge → abort training.

Features:
- Dynamic slippage (base + k * atr_percent)
- Spread widening in volatile candles
- Partial fill simulation
- Intrabar stop/target evaluation
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FillType(Enum):
    """Order fill types"""
    FULL = "full"
    PARTIAL = "partial"
    REJECTED = "rejected"
    DELAYED = "delayed"


class ExitReason(Enum):
    """Trade exit reasons"""
    STOP_LOSS = "stop_loss"
    TARGET = "target"
    TIMEOUT = "timeout"
    SIGNAL = "signal"
    EXPECTANCY = "expectancy"  # V3: Exit when expectancy goes negative


@dataclass
class SlippageModel:
    """
    Dynamic slippage model: slippage = base + k * atr_percent
    
    In volatile candles, slippage increases proportionally.
    """
    base_slippage: float = 0.0005  # 0.05% base
    volatility_factor: float = 0.5  # k factor
    max_slippage: float = 0.02  # Cap at 2%
    
    def calculate(self, atr: float, price: float) -> float:
        """Calculate dynamic slippage based on ATR"""
        if price <= 0 or atr <= 0:
            return self.base_slippage
        
        atr_percent = atr / price
        slippage = self.base_slippage + (self.volatility_factor * atr_percent)
        return min(slippage, self.max_slippage)


@dataclass
class FillResult:
    """Result of an order fill attempt"""
    fill_type: FillType
    fill_price: float
    fill_qty: int
    slippage_applied: float
    delay_bars: int = 0


@dataclass
class ExitResult:
    """Result of exit evaluation"""
    should_exit: bool
    exit_reason: Optional[ExitReason]
    exit_price: float
    bar_idx: int


class ExecutionSimulator:
    """
    Unified trade execution simulator.
    
    CRITICAL: This class must be used by:
    - labeler.py (for label generation)
    - backtesting.py (for backtest simulation)
    - paper_trader.py (for paper trading)
    
    Any divergence in execution logic will cause training/reality mismatch.
    """
    
    VERSION = "3.0.0"  # Bump on any logic change
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_rate: float = 0.0003,  # 0.03%
        enable_partial_fills: bool = False,
        partial_fill_min: float = 0.5,  # Minimum 50% fill
        rejection_rate: float = 0.0,  # 0% rejection by default
        max_delay_bars: int = 0  # No delay by default
    ):
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_rate = commission_rate
        self.enable_partial_fills = enable_partial_fills
        self.partial_fill_min = partial_fill_min
        self.rejection_rate = rejection_rate
        self.max_delay_bars = max_delay_bars
    
    def get_entry_price(
        self,
        bar: pd.Series,
        direction: Literal[1, -1],  # 1=long, -1=short
        atr: float
    ) -> FillResult:
        """
        Calculate entry fill price for NEXT bar.
        
        Entry happens at next bar's OPEN with slippage.
        Long: open * (1 + slippage)
        Short: open * (1 - slippage)
        
        Args:
            bar: The entry bar (next bar after signal)
            direction: 1 for long, -1 for short
            atr: ATR at signal time for slippage calculation
            
        Returns:
            FillResult with fill price and details
        """
        open_price = bar['open']
        
        # Check for rejection (simulates order rejection in volatile markets)
        if self.rejection_rate > 0 and np.random.random() < self.rejection_rate:
            return FillResult(
                fill_type=FillType.REJECTED,
                fill_price=0.0,
                fill_qty=0,
                slippage_applied=0.0
            )
        
        # Calculate dynamic slippage
        slippage = self.slippage_model.calculate(atr, open_price)
        
        # Apply slippage directionally
        if direction == 1:  # Long
            fill_price = open_price * (1 + slippage)
        else:  # Short
            fill_price = open_price * (1 - slippage)
        
        # Determine fill type
        fill_type = FillType.FULL
        fill_ratio = 1.0
        
        if self.enable_partial_fills:
            fill_ratio = np.random.uniform(self.partial_fill_min, 1.0)
            if fill_ratio < 1.0:
                fill_type = FillType.PARTIAL
        
        return FillResult(
            fill_type=fill_type,
            fill_price=round(fill_price, 2),
            fill_qty=int(fill_ratio * 100),  # Percentage as qty placeholder
            slippage_applied=slippage
        )
    
    def get_exit_price(
        self,
        bar: pd.Series,
        direction: Literal[1, -1],
        atr: float
    ) -> float:
        """
        Calculate exit fill price.
        
        Exit happens at bar's CLOSE with slippage (opposite direction).
        Long exit: close * (1 - slippage)
        Short exit: close * (1 + slippage)
        """
        close_price = bar['close']
        slippage = self.slippage_model.calculate(atr, close_price)
        
        if direction == 1:  # Exiting long = selling
            return round(close_price * (1 - slippage), 2)
        else:  # Exiting short = buying back
            return round(close_price * (1 + slippage), 2)
    
    def evaluate_exit_intrabar(
        self,
        bar: pd.Series,
        entry_price: float,
        stop_price: float,
        target_price: float,
        direction: Literal[1, -1],
        bar_idx: int
    ) -> ExitResult:
        """
        Evaluate if stop or target is hit WITHIN a bar using high/low.
        
        This is the CRITICAL function for honest backtesting.
        Must check stop BEFORE target to avoid false wins.
        
        Args:
            bar: Current bar with OHLC
            entry_price: Entry price of the trade
            stop_price: Stop loss price
            target_price: Target price
            direction: 1 for long, -1 for short
            bar_idx: Current bar index
            
        Returns:
            ExitResult with exit details
        """
        high = bar['high']
        low = bar['low']
        
        if direction == 1:  # Long trade
            # Stop hit if low goes below stop
            stop_hit = low <= stop_price
            # Target hit if high goes above target
            target_hit = high >= target_price
            
            if stop_hit and target_hit:
                # Both hit in same bar - assume stop hit first (conservative)
                return ExitResult(
                    should_exit=True,
                    exit_reason=ExitReason.STOP_LOSS,
                    exit_price=stop_price,
                    bar_idx=bar_idx
                )
            elif stop_hit:
                return ExitResult(
                    should_exit=True,
                    exit_reason=ExitReason.STOP_LOSS,
                    exit_price=stop_price,
                    bar_idx=bar_idx
                )
            elif target_hit:
                return ExitResult(
                    should_exit=True,
                    exit_reason=ExitReason.TARGET,
                    exit_price=target_price,
                    bar_idx=bar_idx
                )
        
        else:  # Short trade
            # Stop hit if high goes above stop
            stop_hit = high >= stop_price
            # Target hit if low goes below target
            target_hit = low <= target_price
            
            if stop_hit and target_hit:
                # Both hit in same bar - assume stop hit first (conservative)
                return ExitResult(
                    should_exit=True,
                    exit_reason=ExitReason.STOP_LOSS,
                    exit_price=stop_price,
                    bar_idx=bar_idx
                )
            elif stop_hit:
                return ExitResult(
                    should_exit=True,
                    exit_reason=ExitReason.STOP_LOSS,
                    exit_price=stop_price,
                    bar_idx=bar_idx
                )
            elif target_hit:
                return ExitResult(
                    should_exit=True,
                    exit_reason=ExitReason.TARGET,
                    exit_price=target_price,
                    bar_idx=bar_idx
                )
        
        # No exit
        return ExitResult(
            should_exit=False,
            exit_reason=None,
            exit_price=0.0,
            bar_idx=bar_idx
        )
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        qty: int,
        direction: Literal[1, -1]
    ) -> float:
        """
        Calculate P&L including commission.
        
        Long: (exit - entry) * qty - commission
        Short: (entry - exit) * qty - commission
        """
        if direction == 1:  # Long
            gross_pnl = (exit_price - entry_price) * qty
        else:  # Short
            gross_pnl = (entry_price - exit_price) * qty
        
        # Commission on both sides
        commission = (entry_price * qty + exit_price * qty) * self.commission_rate
        
        return round(gross_pnl - commission, 2)
    
    def calculate_r_multiple(
        self,
        entry_price: float,
        exit_price: float,
        stop_price: float,
        direction: Literal[1, -1]
    ) -> float:
        """
        Calculate R-multiple (risk-adjusted return).
        
        R = actual_move / risk_distance
        """
        if direction == 1:  # Long
            risk = entry_price - stop_price
            reward = exit_price - entry_price
        else:  # Short
            risk = stop_price - entry_price
            reward = entry_price - exit_price
        
        if risk <= 0:
            return 0.0
        
        return round(reward / risk, 2)
    
    def simulate_trade(
        self,
        df: pd.DataFrame,
        signal_bar_idx: int,
        direction: Literal[1, -1],
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 1.5,
        time_limit_bars: int = 24
    ) -> Dict:
        """
        Simulate a complete trade from signal to exit.
        
        This is the FULL trade simulation used by labeler and backtester.
        
        Args:
            df: DataFrame with OHLC and 'atr_14' column
            signal_bar_idx: Index of the signal bar
            direction: 1 for long, -1 for short
            stop_atr_mult: Stop loss multiplier
            target_atr_mult: Target multiplier
            time_limit_bars: Maximum bars to hold
            
        Returns:
            Dict with trade details including outcome
        """
        # Validate we have next bar for entry
        if signal_bar_idx + 1 >= len(df):
            return {'outcome': None, 'reason': 'no_next_bar'}
        
        # Get ATR at signal time
        atr = df.iloc[signal_bar_idx].get('atr_14', df.iloc[signal_bar_idx].get('atr', 1.0))
        
        # Entry at NEXT bar open
        entry_bar = df.iloc[signal_bar_idx + 1]
        entry_result = self.get_entry_price(entry_bar, direction, atr)
        
        if entry_result.fill_type == FillType.REJECTED:
            return {'outcome': None, 'reason': 'rejected'}
        
        entry_price = entry_result.fill_price
        entry_bar_idx = signal_bar_idx + 1
        
        # Calculate stop and target
        if direction == 1:  # Long
            stop_price = entry_price - (atr * stop_atr_mult)
            target_price = entry_price + (atr * target_atr_mult)
        else:  # Short
            stop_price = entry_price + (atr * stop_atr_mult)
            target_price = entry_price - (atr * target_atr_mult)
        
        # Simulate bar by bar
        exit_result = None
        for i in range(entry_bar_idx + 1, min(entry_bar_idx + time_limit_bars + 1, len(df))):
            bar = df.iloc[i]
            exit_result = self.evaluate_exit_intrabar(
                bar, entry_price, stop_price, target_price, direction, i
            )
            if exit_result.should_exit:
                break
        
        # Handle timeout
        if exit_result is None or not exit_result.should_exit:
            if entry_bar_idx + time_limit_bars < len(df):
                timeout_bar = df.iloc[entry_bar_idx + time_limit_bars]
                exit_price = self.get_exit_price(timeout_bar, direction, atr)
                return {
                    'outcome': -1,  # Timeout
                    'reason': 'timeout',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_bar_idx': entry_bar_idx,
                    'exit_bar_idx': entry_bar_idx + time_limit_bars,
                    'r_multiple': self.calculate_r_multiple(entry_price, exit_price, stop_price, direction),
                    'slippage': entry_result.slippage_applied
                }
            else:
                return {'outcome': None, 'reason': 'insufficient_data'}
        
        # Win or loss
        if exit_result.exit_reason == ExitReason.TARGET:
            outcome = 1  # Win
        else:
            outcome = 0  # Loss
        
        return {
            'outcome': outcome,
            'reason': exit_result.exit_reason.value,
            'entry_price': entry_price,
            'exit_price': exit_result.exit_price,
            'entry_bar_idx': entry_bar_idx,
            'exit_bar_idx': exit_result.bar_idx,
            'stop_price': stop_price,
            'target_price': target_price,
            'r_multiple': self.calculate_r_multiple(
                entry_price, exit_result.exit_price, stop_price, direction
            ),
            'slippage': entry_result.slippage_applied
        }


# =============================================================================
# VERSION CHECK - Ensures all consumers use same version
# =============================================================================

def get_simulator_version() -> str:
    """Get current simulator version for consistency checks"""
    return ExecutionSimulator.VERSION


def verify_simulator_version(expected_version: str) -> bool:
    """
    Verify simulator version matches expected.
    
    Call this at training start to abort if versions mismatch.
    """
    current = get_simulator_version()
    if current != expected_version:
        logger.error(f"❌ ExecutionSimulator version mismatch! "
                    f"Expected {expected_version}, got {current}")
        return False
    return True


# =============================================================================
# FACTORY FUNCTIONS for different use cases
# =============================================================================

def get_labeler_simulator() -> ExecutionSimulator:
    """
    Get simulator configured for labeling.
    
    Conservative settings - no partial fills, no delays.
    """
    return ExecutionSimulator(
        slippage_model=SlippageModel(
            base_slippage=0.0005,
            volatility_factor=0.5
        ),
        commission_rate=0.0003,
        enable_partial_fills=False,
        rejection_rate=0.0
    )


def get_backtest_simulator(realistic: bool = True) -> ExecutionSimulator:
    """
    Get simulator configured for backtesting.
    
    Args:
        realistic: If True, enable partial fills and rejection
    """
    if realistic:
        return ExecutionSimulator(
            slippage_model=SlippageModel(
                base_slippage=0.0005,
                volatility_factor=0.5
            ),
            commission_rate=0.0003,
            enable_partial_fills=True,
            partial_fill_min=0.7,
            rejection_rate=0.02  # 2% rejection rate
        )
    else:
        return get_labeler_simulator()


def get_paper_trader_simulator() -> ExecutionSimulator:
    """
    Get simulator configured for paper trading.
    
    Most realistic settings.
    """
    return ExecutionSimulator(
        slippage_model=SlippageModel(
            base_slippage=0.001,  # Higher base for paper
            volatility_factor=0.7
        ),
        commission_rate=0.0003,
        enable_partial_fills=True,
        partial_fill_min=0.5,
        rejection_rate=0.05,  # 5% rejection
        max_delay_bars=1
    )
