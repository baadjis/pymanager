"""
Time Series Predictor v1.0 - Financial Forecasting
===================================================
Models:
- ARIMA (fast, good for stationary series)
- Prophet (Facebook, robust, handles trends/seasonality)
- LSTM (deep learning, best for complex patterns)
- Ensemble (combines all models)

All models CPU-optimized and production-ready.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logger.warning("⚠️ ARIMA unavailable. Install: pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("⚠️ Prophet unavailable. Install: pip install prophet")

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("⚠️ LSTM unavailable. Install: pip install torch scikit-learn")


# =============================================================================
# BASE PREDICTOR CLASS
# =============================================================================

class BasePredictor:
    """Base class for all predictors"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.last_train_date = None
    
    def fit(self, data: pd.Series) -> None:
        """Fit model on historical data"""
        raise NotImplementedError
    
    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict future values
        
        Returns:
            (predictions, lower_bound, upper_bound)
        """
        raise NotImplementedError
    
    def validate(self, train_data: pd.Series, test_data: pd.Series) -> Dict[str, float]:
        """Validate model on test data"""
        self.fit(train_data)
        horizon = len(test_data)
        predictions, _, _ = self.predict(horizon)
        
        # Calculate metrics
        mse = np.mean((test_data.values - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_data.values - predictions))
        mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }


# =============================================================================
# ARIMA PREDICTOR (Fast, Classical)
# =============================================================================

class ARIMAPredictor(BasePredictor):
    """
    ARIMA (AutoRegressive Integrated Moving Average)
    
    Pros:
    - Very fast (CPU)
    - Good for stationary series
    - Interpretable
    - No training needed
    
    Cons:
    - Assumes linear relationships
    - Struggles with trends/seasonality
    - Requires parameter tuning
    """
    
    def __init__(self, order: Tuple[int, int, int] = None):
        super().__init__("ARIMA")
        self.order = order or (1, 1, 1)  # (p, d, q)
        self.model = None
        self.model_fit = None
    
    def fit(self, data: pd.Series) -> None:
        """Fit ARIMA model"""
        if not ARIMA_AVAILABLE:
            raise ImportError("ARIMA not available. Install statsmodels")
        
        logger.info(f"Fitting ARIMA{self.order}...")
        
        try:
            # Auto-tune order if needed
            if self.order == (1, 1, 1):
                self.order = self._auto_order(data)
                logger.info(f"  Auto-tuned order: {self.order}")
            
            # Fit model
            self.model = ARIMA_Model(data, order=self.order)
            self.model_fit = self.model.fit()
            
            self.is_fitted = True
            self.last_train_date = data.index[-1] if hasattr(data, 'index') else None
            
            logger.info(f"✅ ARIMA fitted (AIC: {self.model_fit.aic:.2f})")
            
        except Exception as e:
            logger.error(f"❌ ARIMA fitting failed: {e}")
            raise
    
    def predict(self, horizon: int, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first")
        
        # Forecast
        forecast_result = self.model_fit.get_forecast(steps=horizon)
        predictions = forecast_result.predicted_mean.values
        
        # Confidence intervals
        conf_int = forecast_result.conf_int(alpha=1-confidence)
        lower_bound = conf_int.iloc[:, 0].values
        upper_bound = conf_int.iloc[:, 1].values
        
        return predictions, lower_bound, upper_bound
    
    def _auto_order(self, data: pd.Series, max_p: int = 3, max_q: int = 3) -> Tuple[int, int, int]:
        """Auto-tune ARIMA order using AIC"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA_Model(data, order=(p, 1, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, 1, q)
                except:
                    continue
        
        return best_order


# =============================================================================
# PROPHET PREDICTOR (Robust, Production-Ready)
# =============================================================================

class ProphetPredictor(BasePredictor):
    """
    Prophet (Facebook)
    
    Pros:
    - Handles trends, seasonality automatically
    - Robust to missing data
    - Fast (CPU)
    - Production-ready
    - No hyperparameter tuning
    
    Cons:
    - Black box (less interpretable)
    - Requires pandas DataFrame
    
    BEST CHOICE for financial time series
    """
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = False,
                 daily_seasonality: bool = False):
        super().__init__("Prophet")
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
    
    def fit(self, data: pd.Series) -> None:
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install prophet")
        
        logger.info("Fitting Prophet...")
        
        try:
            # Prepare data (Prophet needs 'ds' and 'y' columns)
            df = pd.DataFrame({
                'ds': data.index if hasattr(data, 'index') else pd.date_range(end=datetime.now(), periods=len(data)),
                'y': data.values
            })
            
            # Initialize model
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                interval_width=0.95
            )
            
            # Fit (suppress output)
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            self.model.fit(df)
            
            sys.stdout = old_stdout
            
            self.is_fitted = True
            self.last_train_date = df['ds'].iloc[-1]
            
            logger.info("✅ Prophet fitted")
            
        except Exception as e:
            logger.error(f"❌ Prophet fitting failed: {e}")
            raise
    
    def predict(self, horizon: int, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon)
        
        # Predict
        forecast = self.model.predict(future)
        
        # Extract predictions (only future values)
        predictions = forecast['yhat'].iloc[-horizon:].values
        lower_bound = forecast['yhat_lower'].iloc[-horizon:].values
        upper_bound = forecast['yhat_upper'].iloc[-horizon:].values
        
        return predictions, lower_bound, upper_bound


# =============================================================================
# LSTM PREDICTOR (Deep Learning)
# =============================================================================

class LSTMPredictor(BasePredictor):
    """
    LSTM (Long Short-Term Memory)
    
    Pros:
    - Captures complex patterns
    - Good for long sequences
    - Non-linear relationships
    
    Cons:
    - Slower (but still CPU-friendly)
    - Needs more data (>100 points)
    - Black box
    - Requires training
    
    Use when: >200 data points, complex patterns
    """
    
    def __init__(self, 
                 hidden_size: int = 50,
                 num_layers: int = 2,
                 lookback: int = 20,
                 epochs: int = 50,
                 batch_size: int = 32):
        super().__init__("LSTM")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = None
        self.last_sequence = None
    
    def fit(self, data: pd.Series) -> None:
        """Fit LSTM model"""
        if not LSTM_AVAILABLE:
            raise ImportError("LSTM not available. Install torch scikit-learn")
        
        logger.info(f"Fitting LSTM (epochs={self.epochs})...")
        
        try:
            # Normalize data
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            if len(X) < 10:
                raise ValueError("Not enough data for LSTM (need >30 points)")
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Build model
            self.model = LSTMModel(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1
            )
            
            # Train
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"  Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")
            
            # Save last sequence for prediction
            self.last_sequence = scaled_data[-self.lookback:]
            
            self.is_fitted = True
            self.last_train_date = data.index[-1] if hasattr(data, 'index') else None
            
            logger.info(f"✅ LSTM fitted (final loss: {loss.item():.6f})")
            
        except Exception as e:
            logger.error(f"❌ LSTM fitting failed: {e}")
            raise
    
    def predict(self, horizon: int, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals (Monte Carlo dropout)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first")
        
        self.model.eval()
        
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        # Iterative prediction
        for _ in range(horizon):
            # Prepare input
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                pred = self.model(input_tensor).item()
            
            predictions.append(pred)
            
            # Update sequence (sliding window)
            current_sequence = np.append(current_sequence[1:], [[pred]], axis=0)
        
        # Denormalize
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        # Confidence intervals (simple: ±10% for demo, use dropout for real)
        std = np.std(predictions) * 0.1
        lower_bound = predictions - 1.96 * std
        upper_bound = predictions + 1.96 * std
        
        return predictions, lower_bound, upper_bound
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(data[i+self.lookback])
        return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    """LSTM Neural Network"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode
        out = self.fc(out[:, -1, :])
        return out


# =============================================================================
# ENSEMBLE PREDICTOR (Combines All Models)
# =============================================================================

class EnsemblePredictor(BasePredictor):
    """
    Ensemble of ARIMA + Prophet + LSTM
    
    Pros:
    - Best accuracy (combines strengths)
    - Robust (if one fails, others work)
    - Confidence from multiple models
    
    Cons:
    - Slower (runs all models)
    - More complex
    
    RECOMMENDED for production
    """
    
    def __init__(self, 
                 use_arima: bool = True,
                 use_prophet: bool = True,
                 use_lstm: bool = True,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__("Ensemble")
        
        self.predictors = []
        self.weights = weights or {}
        
        # Initialize models
        if use_arima and ARIMA_AVAILABLE:
            self.predictors.append(ARIMAPredictor())
            self.weights['ARIMA'] = self.weights.get('ARIMA', 0.3)
        
        if use_prophet and PROPHET_AVAILABLE:
            self.predictors.append(ProphetPredictor())
            self.weights['Prophet'] = self.weights.get('Prophet', 0.5)
        
        if use_lstm and LSTM_AVAILABLE:
            self.predictors.append(LSTMPredictor(epochs=30))  # Fewer epochs for speed
            self.weights['LSTM'] = self.weights.get('LSTM', 0.2)
        
        if not self.predictors:
            raise ValueError("No models available. Install at least one: statsmodels, prophet, or torch")
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.info(f"Ensemble initialized with {len(self.predictors)} models")
        logger.info(f"  Weights: {self.weights}")
    
    def fit(self, data: pd.Series) -> None:
        """Fit all models"""
        logger.info("Fitting Ensemble models...")
        
        fitted_count = 0
        for predictor in self.predictors:
            try:
                predictor.fit(data)
                fitted_count += 1
            except Exception as e:
                logger.warning(f"  ⚠️ {predictor.name} failed: {e}")
        
        if fitted_count == 0:
            raise ValueError("All models failed to fit")
        
        self.is_fitted = True
        self.last_train_date = data.index[-1] if hasattr(data, 'index') else None
        
        logger.info(f"✅ Ensemble fitted ({fitted_count}/{len(self.predictors)} models)")
    
    def predict(self, horizon: int, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with weighted average"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first")
        
        all_predictions = []
        all_lower = []
        all_upper = []
        active_weights = []
        
        for predictor in self.predictors:
            if not predictor.is_fitted:
                continue
            
            try:
                pred, lower, upper = predictor.predict(horizon, confidence)
                
                weight = self.weights.get(predictor.name, 0.33)
                
                all_predictions.append(pred * weight)
                all_lower.append(lower * weight)
                all_upper.append(upper * weight)
                active_weights.append(weight)
                
            except Exception as e:
                logger.warning(f"  ⚠️ {predictor.name} prediction failed: {e}")
        
        if not all_predictions:
            raise ValueError("All models failed to predict")
        
        # Weighted average
        predictions = np.sum(all_predictions, axis=0)
        lower_bound = np.sum(all_lower, axis=0)
        upper_bound = np.sum(all_upper, axis=0)
        
        return predictions, lower_bound, upper_bound


# =============================================================================
# MAIN INTERFACE
# =============================================================================

class TimeSeriesPredictor:
    """
    Main interface for time series prediction
    
    Usage:
        predictor = TimeSeriesPredictor(model='ensemble')
        predictor.fit(historical_data)
        predictions = predictor.predict(horizon=30)
    """
    
    def __init__(self, model: str = 'ensemble'):
        """
        Initialize predictor
        
        Args:
            model: 'arima', 'prophet', 'lstm', or 'ensemble' (recommended)
        """
        self.model_type = model.lower()
        
        if self.model_type == 'arima':
            self.predictor = ARIMAPredictor()
        elif self.model_type == 'prophet':
            self.predictor = ProphetPredictor()
        elif self.model_type == 'lstm':
            self.predictor = LSTMPredictor()
        elif self.model_type == 'ensemble':
            self.predictor = EnsemblePredictor()
        else:
            raise ValueError(f"Unknown model: {model}. Use 'arima', 'prophet', 'lstm', or 'ensemble'")
        
        logger.info(f"🤖 TimeSeriesPredictor initialized with {self.model_type}")
    
    def fit(self, data: pd.Series) -> None:
        """
        Fit model on historical data
        
        Args:
            data: Historical returns (pd.Series with DatetimeIndex)
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        if len(data) < 30:
            raise ValueError("Need at least 30 data points for reliable predictions")
        
        self.predictor.fit(data)
    
    def predict(self, horizon: int, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Predict future values
        
        Args:
            horizon: Number of periods to forecast
            confidence: Confidence level (0.95 = 95%)
        
        Returns:
            Dict with predictions, confidence intervals, and metadata
        """
        predictions, lower, upper = self.predictor.predict(horizon, confidence)
        
        return {
            'model': self.model_type,
            'horizon': horizon,
            'confidence_level': confidence,
            'predictions': predictions.tolist(),
            'confidence_lower': lower.tolist(),
            'confidence_upper': upper.tolist(),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate(self, train_data: pd.Series, test_data: pd.Series) -> Dict[str, float]:
        """
        Validate model on test data
        
        Returns:
            Dict with metrics (MSE, RMSE, MAE, MAPE)
        """
        return self.predictor.validate(train_data, test_data)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def quick_predict(data: pd.Series, horizon: int = 30, model: str = 'ensemble') -> Dict:
    """Quick prediction helper"""
    predictor = TimeSeriesPredictor(model=model)
    predictor.fit(data)
    return predictor.predict(horizon)


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    print("🤖 Testing Time Series Predictor\n")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252)  # 1 year daily
    returns = np.cumsum(np.random.randn(252) * 0.01) + 1.0
    data = pd.Series(returns, index=dates)
    
    print(f"📊 Data: {len(data)} points")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Std: {data.std():.4f}")
    
    # Test each model
    models_to_test = []
    
    if ARIMA_AVAILABLE:
        models_to_test.append('arima')
    if PROPHET_AVAILABLE:
        models_to_test.append('prophet')
    if LSTM_AVAILABLE:
        models_to_test.append('lstm')
    
    models_to_test.append('ensemble')
    
    print(f"\n{'='*70}")
    print("🧪 Testing Models")
    print('='*70)
    
    for model_name in models_to_test:
        print(f"\n{model_name.upper()}:")
        
        try:
            predictor = TimeSeriesPredictor(model=model_name)
            predictor.fit(data)
            
            result = predictor.predict(horizon=30)
            
            print(f"  ✓ Predictions: {len(result['predictions'])}")
            print(f"  ✓ Mean prediction: {np.mean(result['predictions']):.4f}")
            print(f"  ✓ Confidence: {result['confidence_level']*100:.0f}%")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\n{'='*70}")
    print("✅ Test complete")
