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
from app.services.maker_taker_predictor import MakerTakerPredictor
from app.services.slippage_predictor import SlippagePredictor
from app.services.fee_calculator import FeeCalculator
from app.services.performance_monitor import PerformanceMonitor
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
maker_taker_predictor = MakerTakerPredictor()
slippage_predictor = SlippagePredictor()
performance_monitor = PerformanceMonitor()

async def orderbook_callback(data: dict):
    """Callback function for processing orderbook updates"""
    with performance_monitor.measure("data_processing"):
        symbol = data['symbol']
        if symbol in market_impact_calculators:
            market_impact_calculators[symbol].update_orderbook(data)

@app.get("/api/metrics")
async def get_performance_metrics():
    """Get performance metrics for all monitored components"""
    return performance_monitor.get_all_metrics()

@app.get("/api/fee-tiers")
async def get_fee_tiers():
    """Get available fee tiers and their rates"""
    return {
        tier: {
            "rate": rate,
            "percentage": f"{rate * 100:.3f}%"
        }
        for tier, rate in FeeCalculator.FEE_TIER_MAP.items()
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")

@app.post("/api/simulate", response_model=SimulationResults)
async def simulate_trade(params: SimulationParams):
    """Simulate a trade with given parameters"""
    # Start measuring overall simulation loop
    performance_monitor.start_measurement("simulation_loop")
    
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
    
    # Get current market conditions
    with performance_monitor.measure("data_processing"):
        current_spread = calculator.get_current_spread()
        current_depth = calculator.calculate_market_depth()
        # Use manual volatility if enabled, otherwise use dynamic calculation
        current_volatility = params.manual_volatility if params.use_manual_volatility else calculator.calculate_volatility()
        vwap_deviation = calculator.get_vwap_deviation()
    
    # Calculate market impact
    with performance_monitor.measure("market_impact_calc"):
        market_impact = calculator.almgren_chriss_impact(params.quantity)
    
    # Predict slippage
    with performance_monitor.measure("slippage_prediction"):
        slippage = slippage_predictor.predict_slippage(
            quantity=params.quantity,
            volatility=current_volatility,
            spread=current_spread,
            depth=current_depth,
            vwap_deviation=vwap_deviation,
            timestamp=datetime.utcnow()
        )
    
    # Calculate fees using the fee calculator
    with performance_monitor.measure("fee_calculation"):
        fees = FeeCalculator.calculate_fee(params.quantity, params.fee_tier)
    
    # Predict maker/taker proportion
    maker_taker = maker_taker_predictor.predict_maker_probability(
        size=params.quantity,
        volatility=current_volatility,
        spread=current_spread
    )
    
    # End measuring simulation loop
    performance_monitor.end_measurement("simulation_loop")
    
    # Log metrics periodically
    performance_monitor.log_metrics()
    
    return SimulationResults(
        expected_slippage=slippage,
        expected_fees=fees,
        market_impact=market_impact,
        net_cost=slippage + fees + market_impact,
        maker_taker_proportion=maker_taker,
        internal_latency=performance_monitor.get_metrics("simulation_loop").avg_latency,
        timestamp=datetime.utcnow()
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Measure UI update latency
            with performance_monitor.measure("ui_update"):
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