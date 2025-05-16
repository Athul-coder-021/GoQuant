import asyncio
import logging
from datetime import datetime
from typing import Dict

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models.schemas import SimulationParams, SimulationResults, OrderBook
from app.services.market_impact import MarketImpactCalculator
from app.websocket.client import OrderBookWebSocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trade Simulator")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
market_impact_calculators: Dict[str, MarketImpactCalculator] = {}
websocket_clients: Dict[str, OrderBookWebSocket] = {}

async def orderbook_callback(data: dict):
    """Callback function for processing orderbook updates"""
    symbol = data['symbol']
    if symbol in market_impact_calculators:
        market_impact_calculators[symbol].update_orderbook(data)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")

@app.post("/api/simulate", response_model=SimulationResults)
async def simulate_trade(params: SimulationParams):
    """Simulate a trade with given parameters"""
    symbol = params.symbol
    
    # Initialize calculator if not exists
    if symbol not in market_impact_calculators:
        market_impact_calculators[symbol] = MarketImpactCalculator()
        
        # Initialize WebSocket connection if not exists
        if symbol not in websocket_clients:
            ws_uri = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{symbol}"
            websocket_clients[symbol] = OrderBookWebSocket(
                uri=ws_uri,
                symbol=symbol,
                callback=orderbook_callback
            )
            # Start WebSocket connection in background
            asyncio.create_task(websocket_clients[symbol].start())
    
    calculator = market_impact_calculators[symbol]
    
    # Calculate market impact
    start_time = datetime.utcnow()
    market_impact = calculator.almgren_chriss_impact(params.quantity)
    slippage = calculator.estimate_slippage(params.quantity)
    
    # Calculate fees based on fee tier
    fee_rate = 0.001  # Default fee rate, adjust based on fee tier
    fees = params.quantity * fee_rate
    
    # Calculate maker/taker proportion (simplified)
    maker_taker = 0.3  # Assuming 30% maker, 70% taker
    
    # Calculate processing latency
    latency = (datetime.utcnow() - start_time).total_seconds() * 1000  # in milliseconds
    
    return SimulationResults(
        expected_slippage=slippage,
        expected_fees=fees,
        market_impact=market_impact,
        net_cost=slippage + fees + market_impact,
        maker_taker_proportion=maker_taker,
        internal_latency=latency,
        timestamp=datetime.utcnow()
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send updates every second
            for symbol, calculator in market_impact_calculators.items():
                if calculator.orderbook_history:
                    latest_data = calculator.orderbook_history[-1]
                    await websocket.send_json(latest_data)
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 