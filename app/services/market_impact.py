import numpy as np
from typing import List, Tuple
import pandas as pd
from datetime import datetime, timedelta

class MarketImpactCalculator:
    def __init__(self):
        self.sigma = None  # Market volatility
        self.eta = 2.5e-6  # Market impact coefficient
        self.epsilon = 0.0625  # Fixed cost coefficient
        self.orderbook_history = []
        self.price_history = []
        self.trade_history = []  # For VWAP calculation
        
    def update_orderbook(self, orderbook_data: dict):
        """Update internal state with new orderbook data"""
        self.orderbook_history.append(orderbook_data)
        # Keep only last 100 orderbook states
        if len(self.orderbook_history) > 100:
            self.orderbook_history.pop(0)
            
        # Extract mid price and update price history
        best_bid = float(orderbook_data['bids'][0][0])
        best_ask = float(orderbook_data['asks'][0][0])
        mid_price = (best_bid + best_ask) / 2
        timestamp = datetime.fromisoformat(orderbook_data['timestamp'])
        
        self.price_history.append((timestamp, mid_price))
        
        # Simulate some trades for VWAP calculation (in production we will use real trade data)
        trade_size = np.random.lognormal(0, 1)
        self.trade_history.append((timestamp, mid_price, trade_size))
        
        # Keep only last hour of trade history
        cutoff_time = timestamp - timedelta(hours=1)
        self.trade_history = [(t, p, s) for t, p, s in self.trade_history if t > cutoff_time]

    def calculate_vwap(self) -> float:
        """Calculate Volume Weighted Average Price from recent trades"""
        if not self.trade_history:
            return self.get_mid_price()
            
        total_volume = sum(size for _, _, size in self.trade_history)
        if total_volume == 0:
            return self.get_mid_price()
            
        vwap = sum(price * size for _, price, size in self.trade_history) / total_volume
        return vwap
        
    def get_vwap_deviation(self) -> float:
        """Calculate current price deviation from VWAP"""
        if not self.price_history:
            return 0.0
            
        current_price = self.get_mid_price()
        vwap = self.calculate_vwap()
        
        if vwap == 0:
            return 0.0
            
        return (current_price - vwap) / vwap
        
    def get_mid_price(self) -> float:
        """Get current mid price"""
        if not self.orderbook_history:
            return 0.0
            
        latest_book = self.orderbook_history[-1]
        best_bid = float(latest_book['bids'][0][0])
        best_ask = float(latest_book['asks'][0][0])
        return (best_bid + best_ask) / 2

    def calculate_volatility(self) -> float:
        """Calculate market volatility from price history"""
        if len(self.price_history) < 2:
            return 0.0
            
        prices = pd.Series([p[1] for p in self.price_history])
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns.std() * np.sqrt(252)  #With this I will get Annualized volatility

    def calculate_market_depth(self) -> float:
        """Calculate market depth from current orderbook"""
        if not self.orderbook_history:
            return 0.0
            
        current_book = self.orderbook_history[-1]
        depth = 0.0
        
        # Sum up volume within 1% of mid price
        mid_price = (float(current_book['bids'][0][0]) + float(current_book['asks'][0][0])) / 2
        price_range = mid_price * 0.01  # 1%
        
        for bid_price, bid_qty in current_book['bids']:
            if float(bid_price) > mid_price - price_range:
                depth += float(bid_qty)
                
        for ask_price, ask_qty in current_book['asks']:
            if float(ask_price) < mid_price + price_range:
                depth += float(ask_qty)
                
        return depth

    def get_current_spread(self) -> float:
        """Calculate the current bid-ask spread from the latest orderbook data"""
        if not self.orderbook_history:
            return 0.001  # Default spread if no orderbook data
            
        latest_orderbook = self.orderbook_history[-1]
        if not latest_orderbook.get('asks') or not latest_orderbook.get('bids'):
            return 0.001
            
        best_ask = float(latest_orderbook['asks'][0][0])
        best_bid = float(latest_orderbook['bids'][0][0])
        
        return (best_ask - best_bid) / best_bid  # Return relative spread

    def almgren_chriss_impact(self, quantity: float, timeframe: float = 1.0) -> float:
        """
        Calculate market impact using Almgren-Chriss model
        
        Args:
            quantity: Order size in base currency
            timeframe: Time horizon for execution in hours
            
        Returns:
            Estimated market impact in percentage
        """
        if not self.orderbook_history:
            return 0.0
            
        # Update volatility
        self.sigma = self.calculate_volatility()
        
        # Market depth
        market_depth = self.calculate_market_depth()
        
        # Convert timeframe to days
        T = timeframe / 24.0
        
        # Temporary impact
        temporary_impact = self.epsilon * self.sigma * np.sqrt(T)
        
        # Permanent impact
        permanent_impact = self.eta * quantity / market_depth if market_depth > 0 else 0
        
        # Total impact
        total_impact = temporary_impact + permanent_impact
        
        return total_impact * 100  # Convert to percentage

    def estimate_slippage(self, quantity: float) -> float:
        """Estimate slippage based on orderbook depth"""
        if not self.orderbook_history:
            return 0.0
            
        current_book = self.orderbook_history[-1]
        remaining_quantity = quantity
        total_cost = 0.0
        
        # Calculate mid price
        mid_price = (float(current_book['bids'][0][0]) + float(current_book['asks'][0][0])) / 2
        
        # Simulate market order execution
        for price, qty in current_book['asks']:
            price, qty = float(price), float(qty)
            if remaining_quantity <= 0:
                break
                
            executed_qty = min(remaining_quantity, qty)
            total_cost += executed_qty * price
            remaining_quantity -= executed_qty
            
        # If order couldn't be fully filled with current orderbook
        if remaining_quantity > 0:
            total_cost += remaining_quantity * price * 1.01  # Assume 1% worse price for remaining quantity
            
        # Calculate slippage
        ideal_cost = quantity * mid_price
        slippage = ((total_cost - ideal_cost) / ideal_cost) * 100
        
        return slippage 