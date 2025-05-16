# Trade Simulator

A high-performance trade simulator leveraging real-time market data to estimate transaction costs and market impact.

## Features

- Real-time L2 orderbook data processing
- Transaction cost analysis
- Market impact estimation using Almgren-Chriss model
- Slippage prediction
- Fee calculation
- Maker/Taker proportion estimation
- Latency monitoring

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Project Structure

```
trade_simulator/
├── app/
│   ├── models/           # Data models and schemas
│   ├── services/         # Business logic
│   ├── websocket/        # WebSocket client and handlers
│   └── utils/           # Utility functions
├── static/              # Frontend static files
├── main.py             # Application entry point
└── requirements.txt    # Python dependencies
```

## API Documentation

The API documentation is available at `http://localhost:8000/docs` when the server is running. 