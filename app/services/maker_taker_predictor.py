import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Optional

class MakerTakerPredictor:
    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize or load the logistic regression model"""
        model_path = "models/maker_taker_model.joblib"
        scaler_path = "models/maker_taker_scaler.joblib"

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # Load existing model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            # Initialize new model and scaler
            self.model = LogisticRegression(random_state=42)
            self.scaler = StandardScaler()
            
            # Generate synthetic training data
            # This is a simplified example; in production, you'd use real historical data
            np.random.seed(42)
            n_samples = 10000
            
            # Generate synthetic features
            sizes = np.random.lognormal(0, 1, n_samples)  # Order sizes follow lognormal distribution
            volatilities = np.random.gamma(2, 0.5, n_samples)  # Volatility follows gamma distribution
            spreads = np.random.exponential(0.001, n_samples)  # Spreads follow exponential distribution
            
            X = np.column_stack([sizes, volatilities, spreads])
            
            # Generate synthetic labels based on domain knowledge
            # Higher chance of maker orders when:
            # - Size is larger
            # - Volatility is lower
            # - Spread is wider
            probabilities = 1 / (1 + np.exp(-(
                -0.5 * sizes/np.mean(sizes) +  # Larger sizes -> more likely to be maker
                2 * volatilities/np.mean(volatilities) +  # Higher volatility -> more likely to be taker
                -1 * spreads/np.mean(spreads)  # Wider spread -> more likely to be maker
            )))
            y = (np.random.random(n_samples) < probabilities).astype(int)
            
            # Fit the model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Save the model and scaler
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

    def predict_maker_probability(self, size: float, volatility: float, spread: float) -> float:
        """
        Predict the probability of an order being a maker order.
        
        Args:
            size: Order size in base currency
            volatility: Current market volatility
            spread: Current bid-ask spread
            
        Returns:
            float: Probability of the order being a maker order (0.0 to 1.0)
        """
        if self.model is None or self.scaler is None:
            self._initialize_model()
            
        features = np.array([[size, volatility, spread]])
        features_scaled = self.scaler.transform(features)
        
        # Get probability of being a maker order
        maker_prob = self.model.predict_proba(features_scaled)[0][1]
        return float(maker_prob) 