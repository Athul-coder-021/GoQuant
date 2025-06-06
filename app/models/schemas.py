from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from datetime import datetime

class OrderBookLevel(BaseModel):
    price: float
    quantity: float

class OrderBook(BaseModel):
    timestamp: datetime
    exchange: str
    symbol: str
    asks: List[Tuple[str, str]]
    bids: List[Tuple[str, str]]

class SimulationParams(BaseModel):
    exchange: str = Field(default="OKX")
    symbol: str = Field(description="Trading pair symbol")
    order_type: str = Field(default="market")
    quantity: float = Field(description="Order quantity in USD")
    fee_tier: str = Field(description="Exchange fee tier", default="VIP0")
    manual_volatility: Optional[float] = Field(None, description="Manual volatility override (if enabled)")
    use_manual_volatility: bool = Field(default=False, description="Whether to use manual volatility")

class SimulationResults(BaseModel):
    expected_slippage: float
    expected_fees: float
    market_impact: float
    net_cost: float
    maker_taker_proportion: float
    internal_latency: float
    timestamp: datetime = Field(default_factory=datetime.utcnow) 