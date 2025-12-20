"""
Options trading support with Greeks calculation
"""
import numpy as np
from scipy.stats import norm
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OptionsGreeks:
    """
    Calculate option Greeks using Black-Scholes model
    """
    @staticmethod
    def calculate_greeks(
        S: float,  # Spot price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        r: float = 0.06,  # Risk-free rate
        sigma: float = 0.20,  # Volatility
        option_type: str = 'call'
    ) -> Dict:
        """
        Calculate all Greeks for an option
        
        Returns:
            dict: price, delta, gamma, vega, theta, rho
        """
        if T <= 0:
            # Option expired
            if option_type == 'call':
                return {
                    'price': max(S - K, 0),
                    'delta': 1.0 if S > K else 0.0,
                    'gamma': 0.0,
                    'vega': 0.0,
                    'theta': 0.0,
                    'rho': 0.0
                }
            else:
                return {
                    'price': max(K - S, 0),
                    'delta': -1.0 if K > S else 0.0,
                    'gamma': 0.0,
                    'vega': 0.0,
                    'theta': 0.0,
                    'rho': 0.0
                }
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Calculate price
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
            theta_daily = ((-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) -
                           r*K*np.exp(-r*T)*norm.cdf(d2)) / 365)
            rho = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
        else:  # put
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            theta_daily = ((-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) +
                           r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365)
            rho = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
        
        # Calculate greeks (same for call and put)
        gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T) / 100  # Per 1% change in volatility
        
        return {
            'price': float(price),
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta_daily),
            'rho': float(rho),
            'd1': float(d1),
            'd2': float(d2)
        }
    
    @staticmethod
    def calculate_implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float = 0.06,
        option_type: str = 'call',
        max_iterations: int = 100,
        tolerance: float = 0.001
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        sigma = 0.20  # Initial guess
        
        for i in range(max_iterations):
            greeks = OptionsGreeks.calculate_greeks(S, K, T, r, sigma, option_type)
            price_diff = greeks['price'] - market_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            # Newton-Raphson: sigma_new = sigma - f(sigma)/f'(sigma)
            # f(sigma) = BS_price - market_price
            # f'(sigma) = vega
            vega = greeks['vega'] * 100  # Convert back to full vega
            if vega == 0:
                break
            
            sigma = sigma - price_diff / vega
            
            # Keep sigma positive and reasonable
            sigma = max(0.01, min(sigma, 2.0))
        
        logger.warning(f"IV calculation did not converge after {max_iterations} iterations")
        return sigma


class OptionsStrategy:
    """
    Pre-defined options strategies
    """
    
    @staticmethod
    def iron_condor(
        spot: float,
        wing_width: float = 100,
        body_width: float = 50,
        expiry_days: int = 30
    ) -> Dict:
        """
        Iron Condor: Sell OTM call + put, buy further OTM call + put
        
        Args:
            spot: Current underlying price
            wing_width: Distance between long and short strikes
            body_width: Distance of short strikes from spot
        """
        T = expiry_days / 365
        
        # Define strikes
        short_call_strike = spot + body_width
        long_call_strike = short_call_strike + wing_width
        short_put_strike = spot - body_width
        long_put_strike = short_put_strike - wing_width
        
        # Calculate Greeks for all legs
        greeks_calc = OptionsGreeks()
        
        short_call = greeks_calc.calculate_greeks(spot, short_call_strike, T, option_type='call')
        long_call = greeks_calc.calculate_greeks(spot, long_call_strike, T, option_type='call')
        short_put = greeks_calc.calculate_greeks(spot, short_put_strike, T, option_type='put')
        long_put = greeks_calc.calculate_greeks(spot, long_put_strike, T, option_type='put')
        
        # Net position (short - long)
        net_premium = (short_call['price'] + short_put['price'] -
                      long_call['price'] - long_put['price'])
        
        net_delta = (short_call['delta'] + short_put['delta'] -
                    long_call['delta'] - long_put['delta'])
        
        max_profit = net_premium
        max_loss = wing_width - net_premium
        
        return {
            'strategy': 'iron_condor',
            'legs': {
                'short_call': {'strike': short_call_strike, **short_call},
                'long_call': {'strike': long_call_strike, **long_call},
                'short_put': {'strike': short_put_strike, **short_put},
                'long_put': {'strike': long_put_strike, **long_put}
            },
            'net_premium': net_premium,
            'net_delta': net_delta,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_upper': short_call_strike + net_premium,
            'breakeven_lower': short_put_strike - net_premium,
            'probability_of_profit': 'TBD'  # Requires volatility distribution
        }
    
    @staticmethod
    def bull_call_spread(
        spot: float,
        width: float = 50,
        expiry_days: int = 30
    ) -> Dict:
        """
        Bull Call Spread: Buy ATM call, sell OTM call
        """
        T = expiry_days / 365
        
        long_strike = spot
        short_strike = spot + width
        
        greeks_calc = OptionsGreeks()
        long_call = greeks_calc.calculate_greeks(spot, long_strike, T, option_type='call')
        short_call = greeks_calc.calculate_greeks(spot, short_strike, T, option_type='call')
        
        net_cost = long_call['price'] - short_call['price']
        max_profit = width - net_cost
        max_loss = net_cost
        
        return {
            'strategy': 'bull_call_spread',
            'legs': {
                'long_call': {'strike': long_strike, **long_call},
                'short_call': {'strike': short_strike, **short_call}
            },
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': long_strike + net_cost
        }
    
    @staticmethod
    def straddle(
        spot: float,
        expiry_days: int = 30
    ) -> Dict:
        """
        Long Straddle: Buy ATM call + put
        Profits from large moves in either direction
        """
        T = expiry_days / 365
        strike = spot
        
        greeks_calc = OptionsGreeks()
        call = greeks_calc.calculate_greeks(spot, strike, T, option_type='call')
        put = greeks_calc.calculate_greeks(spot, strike, T, option_type='put')
        
        total_cost = call['price'] + put['price']
        
        return {
            'strategy': 'long_straddle',
            'legs': {
                'call': {'strike': strike, **call},
                'put': {'strike': strike, **put}
            },
            'total_cost': total_cost,
            'max_loss': total_cost,
            'breakeven_upper': strike + total_cost,
            'breakeven_lower': strike - total_cost,
            'net_gamma': call['gamma'] + put['gamma'],
            'net_vega': call['vega'] + put['vega']
        }
