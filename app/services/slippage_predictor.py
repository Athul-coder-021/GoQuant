import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Optional, List, Dict
from datetime import datetime, timedelta

class SlippagePredictor:
    def __init__(self):
        self.model: Optional[LinearRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names = ['quantity', 'volatility', 'spread', 'depth', 'vwap_deviation', 'time_of_day']
        self._initialize_model()

    def _initialize_model(self):
        """Initialize or load the linear regression model"""
        model_path = "models/slippage_model.joblib"
        scaler_path = "models/slippage_scaler.joblib"

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # Load existing model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            # Initialize new model and scaler
            self.model = LinearRegression()
            self.scaler = StandardScaler()
            
            # Generate synthetic training data
            np.random.seed(42)
            n_samples = 10000
            
            # Generate synthetic features
            quantities = np.random.lognormal(0, 1, n_samples)  # Order sizes
            volatilities = np.random.gamma(2, 0.5, n_samples)  # Market volatility
            spreads = np.random.exponential(0.001, n_samples)  # Bid-ask spreads
            depths = np.random.lognormal(10, 1, n_samples)  # Market depth
            vwap_deviations = np.random.normal(0, 0.001, n_samples)  # Deviation from VWAP
            time_of_day = np.random.uniform(0, 24, n_samples)  # Hour of day (0-24)
            
            # Combine features
            X = np.column_stack([
                quantities,
                volatilities,
                spreads,
                depths,
                vwap_deviations,
                time_of_day
            ])
            
            # Generate synthetic slippage based on known relationships
            # Slippage tends to:
            # 1. Increase with order size (square root relationship)
            # 2. Increase with volatility (linear relationship)
            # 3. Increase with spread (linear relationship)
            # 4. Decrease with market depth (inverse relationship)
            # 5. Increase with VWAP deviation
            # 6. Higher during market open/close
            
            base_slippage = (
                0.1 * np.sqrt(quantities/np.mean(quantities)) +  # Square root of normalized quantity
                0.2 * volatilities/np.mean(volatilities) +      # Linear with volatility
                0.3 * spreads/np.mean(spreads) +               # Linear with spread
                0.1 / (depths/np.mean(depths)) +               # Inverse with depth
                0.2 * np.abs(vwap_deviations) +               # Linear with VWAP deviation
                0.1 * np.sin(time_of_day * np.pi / 12)        # Cyclical with time of day
            )
            
            # Add some noise
            noise = np.random.normal(0, 0.001, n_samples)
            y = base_slippage + noise
            
            # Fit the model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Save the model and scaler
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

    def predict_slippage(self, 
                        quantity: float, 
                        volatility: float, 
                        spread: float, 
                        depth: float,
                        vwap_deviation: float,
                        timestamp: datetime = None) -> float:
        """
        Predict expected slippage using the regression model.
        
        Args:
            quantity: Order size in base currency
            volatility: Current market volatility
            spread: Current bid-ask spread
            depth: Current market depth
            vwap_deviation: Current price deviation from VWAP
            timestamp: Current timestamp (for time of day feature)
            
        Returns:
            float: Predicted slippage as a percentage
        """
        if self.model is None or self.scaler is None:
            self._initialize_model()
            
        # Get time of day if timestamp provided
        if timestamp is None:
            timestamp = datetime.utcnow()
        time_of_day = timestamp.hour + timestamp.minute / 60.0
            
        # Prepare features
        features = np.array([[
            quantity,
            volatility,
            spread,
            depth,
            vwap_deviation,
            time_of_day
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict slippage
        slippage = self.model.predict(features_scaled)[0]
        
        # Convert to percentage and ensure non-negative
        return slippage