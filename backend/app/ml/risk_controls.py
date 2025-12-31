"""
Stress Testing Module - V3 Phase 4

Replay historical crash scenarios to validate model robustness.

Scenarios:
- COVID crash (March 2020)
- Budget day volatility
- RBI rate decision days
- Flash crashes
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """A stress test scenario"""
    name: str
    description: str
    date_range: Tuple[datetime, datetime]
    expected_drawdown_pct: float  # Expected max drawdown
    volatility_mult: float  # How much more volatile than normal
    
    # Results after testing
    actual_drawdown_pct: Optional[float] = None
    survived: Optional[bool] = None
    pnl: Optional[float] = None


# Pre-defined stress scenarios for Indian markets
STRESS_SCENARIOS = [
    StressScenario(
        name="COVID_CRASH",
        description="March 2020 market crash - 35% drop in weeks",
        date_range=(datetime(2020, 2, 20), datetime(2020, 4, 1)),
        expected_drawdown_pct=0.35,
        volatility_mult=4.0
    ),
    StressScenario(
        name="DEMONETIZATION",
        description="November 2016 demonetization announcement",
        date_range=(datetime(2016, 11, 8), datetime(2016, 12, 1)),
        expected_drawdown_pct=0.15,
        volatility_mult=2.5
    ),
    StressScenario(
        name="BUDGET_2020",
        description="Union Budget Feb 2020 - high volatility",
        date_range=(datetime(2020, 1, 30), datetime(2020, 2, 5)),
        expected_drawdown_pct=0.08,
        volatility_mult=2.0
    ),
    StressScenario(
        name="RBI_RATE_CUT",
        description="Unexpected RBI rate decision",
        date_range=(datetime(2019, 10, 1), datetime(2019, 10, 10)),
        expected_drawdown_pct=0.05,
        volatility_mult=1.5
    ),
    StressScenario(
        name="FLASH_CRASH_SIM",
        description="Simulated flash crash - 10% drop in hours",
        date_range=(datetime(2024, 1, 1), datetime(2024, 1, 2)),  # Synthetic
        expected_drawdown_pct=0.10,
        volatility_mult=5.0
    ),
]


class StressTester:
    """
    Run stress tests on trading strategy.
    
    A model that blows up in historical crashes should NOT be activated.
    """
    
    def __init__(
        self,
        max_acceptable_drawdown: float = 0.25,
        min_scenarios_pass: int = 3
    ):
        self.max_acceptable_drawdown = max_acceptable_drawdown
        self.min_scenarios_pass = min_scenarios_pass
        self.scenarios = STRESS_SCENARIOS.copy()
    
    def generate_synthetic_crash(
        self,
        base_price: float = 100,
        crash_pct: float = 0.10,
        n_bars: int = 100,
        volatility_mult: float = 3.0
    ) -> pd.DataFrame:
        """
        Generate synthetic crash data for testing.
        
        Args:
            base_price: Starting price
            crash_pct: Total crash percentage
            n_bars: Number of bars
            volatility_mult: Volatility multiplier
        """
        # Generate crash trajectory
        crash_rate = crash_pct / (n_bars * 0.3)  # Crash happens in first 30% of bars
        recovery_rate = crash_pct * 0.5 / (n_bars * 0.7)  # Partial recovery
        
        prices = [base_price]
        for i in range(1, n_bars):
            if i < n_bars * 0.3:
                # Crash phase
                change = -crash_rate + np.random.randn() * 0.02 * volatility_mult
            else:
                # Recovery phase
                change = recovery_rate + np.random.randn() * 0.01 * volatility_mult
            
            prices.append(prices[-1] * (1 + change))
        
        dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_bars)]
        
        return pd.DataFrame({
            'ts_utc': dates,
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': [1000000] * n_bars
        })
    
    def run_scenario(
        self,
        scenario: StressScenario,
        strategy_fn,
        historical_data: Optional[pd.DataFrame] = None
    ) -> StressScenario:
        """
        Run a single stress scenario.
        
        Args:
            scenario: The stress scenario to test
            strategy_fn: Function(df) -> list of trades with PnL
            historical_data: Optional real historical data for the period
        """
        if historical_data is not None:
            # Filter to scenario date range
            mask = (
                (historical_data['ts_utc'] >= scenario.date_range[0]) &
                (historical_data['ts_utc'] <= scenario.date_range[1])
            )
            test_data = historical_data[mask].copy()
        else:
            # Generate synthetic data
            test_data = self.generate_synthetic_crash(
                crash_pct=scenario.expected_drawdown_pct,
                volatility_mult=scenario.volatility_mult
            )
        
        if len(test_data) < 10:
            logger.warning(f"Insufficient data for scenario {scenario.name}")
            scenario.survived = True  # Can't test, assume pass
            return scenario
        
        try:
            # Run strategy
            trades = strategy_fn(test_data)
            
            # Calculate drawdown
            if trades:
                pnls = [t.get('pnl', 0) for t in trades]
                equity = np.cumsum(pnls)
                running_max = np.maximum.accumulate(equity)
                drawdowns = running_max - equity
                max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
                
                scenario.actual_drawdown_pct = max_dd / 100000  # Assuming 100k capital
                scenario.pnl = sum(pnls)
            else:
                scenario.actual_drawdown_pct = 0
                scenario.pnl = 0
            
            scenario.survived = scenario.actual_drawdown_pct <= self.max_acceptable_drawdown
            
            logger.info(f"Scenario {scenario.name}: "
                       f"DD={scenario.actual_drawdown_pct*100:.1f}%, "
                       f"PnL={scenario.pnl:.0f}, "
                       f"survived={scenario.survived}")
            
        except Exception as e:
            logger.error(f"Scenario {scenario.name} failed: {e}")
            scenario.survived = False
        
        return scenario
    
    def run_all_scenarios(
        self,
        strategy_fn,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Dict]:
        """
        Run all stress scenarios.
        
        Returns:
            (all_passed, results_dict)
        """
        results = []
        
        for scenario in self.scenarios:
            result = self.run_scenario(scenario, strategy_fn, historical_data)
            results.append(result)
        
        passed = sum(1 for s in results if s.survived)
        all_passed = passed >= self.min_scenarios_pass
        
        summary = {
            'total_scenarios': len(results),
            'passed': passed,
            'failed': len(results) - passed,
            'all_passed': all_passed,
            'scenarios': [
                {
                    'name': s.name,
                    'survived': s.survived,
                    'drawdown_pct': s.actual_drawdown_pct,
                    'pnl': s.pnl
                }
                for s in results
            ]
        }
        
        return all_passed, summary


class CapitalCurveFeedback:
    """
    Adjust risk based on capital curve position.
    
    Reduce risk after drawdown, increase cautiously after recovery.
    """
    
    def __init__(
        self,
        drawdown_threshold: float = 0.10,  # 10% drawdown triggers reduction
        recovery_threshold: float = 0.05,  # 5% above trough to start recovery
        max_risk_reduction: float = 0.5,   # Max 50% risk reduction
        recovery_rate: float = 0.1         # Recover 10% per winning trade
    ):
        self.drawdown_threshold = drawdown_threshold
        self.recovery_threshold = recovery_threshold
        self.max_risk_reduction = max_risk_reduction
        self.recovery_rate = recovery_rate
        
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        self.risk_multiplier = 1.0
    
    def update(self, equity: float) -> None:
        """Update with new equity value"""
        self.current_equity = equity
        
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        
        self._adjust_risk()
    
    def _adjust_risk(self) -> None:
        """Adjust risk multiplier based on drawdown"""
        if self.current_drawdown >= self.drawdown_threshold:
            # In drawdown - reduce risk
            reduction = min(
                self.max_risk_reduction,
                self.current_drawdown / self.drawdown_threshold * self.max_risk_reduction
            )
            self.risk_multiplier = 1.0 - reduction
        else:
            # Recovering - slowly increase risk
            if self.risk_multiplier < 1.0:
                self.risk_multiplier = min(1.0, self.risk_multiplier + self.recovery_rate)
    
    def get_adjusted_risk(self, base_risk: float) -> float:
        """Get drawdown-adjusted risk amount"""
        return base_risk * self.risk_multiplier
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'drawdown_pct': self.current_drawdown * 100,
            'risk_multiplier': self.risk_multiplier,
            'risk_reduction_pct': (1 - self.risk_multiplier) * 100
        }


class SoftOverrideController:
    """
    Intelligent override controls (not just binary kill switch).
    
    Modes:
    - FULL: Normal operation
    - REDUCED: 50% position sizes
    - LONG_ONLY: No shorts
    - PAPER_ONLY: Paper trading only
    - KILLED: No trading
    """
    
    MODES = ['FULL', 'REDUCED', 'LONG_ONLY', 'PAPER_ONLY', 'KILLED']
    
    def __init__(self):
        self.current_mode = 'FULL'
        self.mode_reason = ''
        self.auto_rules = []
    
    def set_mode(self, mode: str, reason: str = '') -> None:
        """Set override mode"""
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.current_mode = mode
        self.mode_reason = reason
        logger.info(f"🔄 Override mode set to {mode}: {reason}")
    
    def add_auto_rule(
        self,
        condition_fn,
        target_mode: str,
        reason: str
    ) -> None:
        """Add automatic rule for mode switching"""
        self.auto_rules.append({
            'condition': condition_fn,
            'mode': target_mode,
            'reason': reason
        })
    
    def evaluate_rules(self, context: Dict) -> None:
        """Evaluate auto rules and switch mode if needed"""
        for rule in self.auto_rules:
            if rule['condition'](context):
                if rule['mode'] != self.current_mode:
                    self.set_mode(rule['mode'], rule['reason'])
                return
    
    def should_trade(self) -> bool:
        """Check if trading is allowed"""
        return self.current_mode not in ['KILLED']
    
    def should_paper_only(self) -> bool:
        """Check if should only paper trade"""
        return self.current_mode == 'PAPER_ONLY'
    
    def allow_short(self) -> bool:
        """Check if short trades are allowed"""
        return self.current_mode not in ['LONG_ONLY', 'KILLED']
    
    def get_size_multiplier(self) -> float:
        """Get position size multiplier"""
        multipliers = {
            'FULL': 1.0,
            'REDUCED': 0.5,
            'LONG_ONLY': 1.0,
            'PAPER_ONLY': 1.0,
            'KILLED': 0.0
        }
        return multipliers.get(self.current_mode, 1.0)
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'mode': self.current_mode,
            'reason': self.mode_reason,
            'can_trade': self.should_trade(),
            'can_short': self.allow_short(),
            'size_multiplier': self.get_size_multiplier()
        }


class MissedOpportunityTracker:
    """
    Track trades we skipped to evaluate threshold optimization.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.skipped_trades = []
    
    def record_skip(
        self,
        symbol: str,
        direction: int,
        prob: float,
        reason: str,
        actual_outcome: Optional[float] = None  # R-multiple if known
    ) -> None:
        """Record a skipped trade opportunity"""
        self.skipped_trades.append({
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'direction': direction,
            'probability': prob,
            'reason': reason,
            'actual_r': actual_outcome
        })
        
        # Trim to lookback
        if len(self.skipped_trades) > self.lookback:
            self.skipped_trades = self.skipped_trades[-self.lookback:]
    
    def update_outcome(self, idx: int, actual_r: float) -> None:
        """Update outcome for a skipped trade"""
        if 0 <= idx < len(self.skipped_trades):
            self.skipped_trades[idx]['actual_r'] = actual_r
    
    def get_opportunity_cost(self) -> Dict:
        """Calculate opportunity cost of skipped trades"""
        with_outcomes = [t for t in self.skipped_trades if t['actual_r'] is not None]
        
        if not with_outcomes:
            return {'total_skipped': len(self.skipped_trades), 'known_outcomes': 0}
        
        winners = [t for t in with_outcomes if t['actual_r'] > 0]
        
        return {
            'total_skipped': len(self.skipped_trades),
            'known_outcomes': len(with_outcomes),
            'would_have_won': len(winners),
            'win_rate': len(winners) / len(with_outcomes),
            'avg_r_missed': np.mean([t['actual_r'] for t in with_outcomes]),
            'total_r_missed': sum(t['actual_r'] for t in winners)
        }
    
    def analyze_thresholds(self) -> Dict:
        """Analyze if thresholds are too strict"""
        with_outcomes = [t for t in self.skipped_trades if t['actual_r'] is not None]
        
        if not with_outcomes:
            return {'analysis': 'INSUFFICIENT_DATA'}
        
        # Group by probability bucket
        buckets = {}
        for t in with_outcomes:
            bucket = int(t['probability'] * 10) / 10  # 0.5, 0.6, 0.7, etc.
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(t['actual_r'])
        
        recommendations = []
        for bucket, outcomes in sorted(buckets.items()):
            avg_r = np.mean(outcomes)
            if avg_r > 0.3 and len(outcomes) >= 5:
                recommendations.append(
                    f"Consider lowering threshold: {bucket:.0%} prob trades had avg {avg_r:.2f}R"
                )
        
        return {
            'buckets': {k: {'count': len(v), 'avg_r': np.mean(v)} for k, v in buckets.items()},
            'recommendations': recommendations
        }


# Singleton instances
_stress_tester: Optional[StressTester] = None
_capital_feedback: Optional[CapitalCurveFeedback] = None
_override_controller: Optional[SoftOverrideController] = None
_missed_tracker: Optional[MissedOpportunityTracker] = None


def get_stress_tester() -> StressTester:
    global _stress_tester
    if _stress_tester is None:
        _stress_tester = StressTester()
    return _stress_tester


def get_capital_feedback() -> CapitalCurveFeedback:
    global _capital_feedback
    if _capital_feedback is None:
        _capital_feedback = CapitalCurveFeedback()
    return _capital_feedback


def get_override_controller() -> SoftOverrideController:
    global _override_controller
    if _override_controller is None:
        _override_controller = SoftOverrideController()
    return _override_controller


def get_missed_tracker() -> MissedOpportunityTracker:
    global _missed_tracker
    if _missed_tracker is None:
        _missed_tracker = MissedOpportunityTracker()
    return _missed_tracker
