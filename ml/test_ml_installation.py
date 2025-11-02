"""
PyManager ML Module v2.0 - Installation & Validation Test
==========================================================

Automated testing script to verify:
1. All ML libraries are installed correctly
2. Models can be initialized
3. Predictions work
4. Performance metrics are calculated

Run this after installing dependencies.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Color output
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = RED = YELLOW = CYAN = BLUE = MAGENTA = ""
    class Style:
        RESET_ALL = BRIGHT = ""

# =============================================================================
# TEST FRAMEWORK
# =============================================================================

class TestRunner:
    """Simple test runner with colored output"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.results = []
    
    def test(self, name: str, func, *args, **kwargs):
        """Run a test and record result"""
        print(f"\n{Fore.CYAN}Testing: {name}{Style.RESET_ALL}")
        
        try:
            result = func(*args, **kwargs)
            
            if result is False:
                self._fail(name, "Test returned False")
            else:
                self._pass(name, result)
                
        except Exception as e:
            self._fail(name, str(e))
    
    def _pass(self, name: str, info: str = ""):
        self.passed += 1
        msg = f"{Fore.GREEN}✅ PASS{Style.RESET_ALL} - {name}"
        if info:
            msg += f"\n       {Fore.BLUE}{info}{Style.RESET_ALL}"
        print(msg)
        self.results.append(('PASS', name))
    
    def _fail(self, name: str, error: str):
        self.failed += 1
        msg = f"{Fore.RED}❌ FAIL{Style.RESET_ALL} - {name}\n       {Fore.RED}Error: {error}{Style.RESET_ALL}"
        print(msg)
        self.results.append(('FAIL', name))
    
    def warn(self, message: str):
        self.warnings += 1
        print(f"{Fore.YELLOW}⚠️  WARNING{Style.RESET_ALL} - {message}")
    
    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*80)
        print(f"{Fore.CYAN}TEST SUMMARY{Style.RESET_ALL}")
        print("="*80)
        
        print(f"\n{Fore.GREEN}Passed: {self.passed}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {self.failed}{Style.RESET_ALL}")
        if self.warnings > 0:
            print(f"{Fore.YELLOW}Warnings: {self.warnings}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Success Rate: {success_rate:.1f}%{Style.RESET_ALL}")
        
        if self.failed == 0:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 ALL TESTS PASSED!{Style.RESET_ALL}")
            return 0
        else:
            print(f"\n{Fore.RED}{Style.BRIGHT}❌ SOME TESTS FAILED{Style.RESET_ALL}")
            return 1


# =============================================================================
# TESTS
# =============================================================================

def test_imports(runner: TestRunner):
    """Test 1: Check if all libraries can be imported"""
    
    libraries = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'statsmodels': 'statsmodels',
        'prophet': 'prophet',
        'torch': 'torch',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost'
    }
    
    available = []
    missing = []
    
    for display_name, import_name in libraries.items():
        try:
            __import__(import_name)
            available.append(display_name)
        except ImportError:
            missing.append(display_name)
    
    # Core libraries must be present
    core = ['numpy', 'pandas', 'scikit-learn']
    missing_core = [lib for lib in core if lib in missing]
    
    if missing_core:
        return False
    
    # Warn about missing optional libraries
    if missing:
        for lib in missing:
            runner.warn(f"{lib} not installed (optional)")
    
    return f"{len(available)}/{len(libraries)} libraries available: {', '.join(available)}"


def test_ml_module_import(runner: TestRunner):
    """Test 2: Import ML module"""
    
    try:
        from ml.timeseries_predictors import TimeSeriesPredictor
        return "ML module imported successfully"
    except ImportError as e:
        if "No module named 'ml'" in str(e):
            # Try current directory
            sys.path.insert(0, '.')
            from ml.timeseries_predictors import TimeSeriesPredictor
            return "ML module imported (from current dir)"
        raise


def test_list_models(runner: TestRunner):
    """Test 3: List available models"""
    
    from ml.timeseries_predictors import TimeSeriesPredictor
    
    available = TimeSeriesPredictor.list_available_models()
    available_models = [name for name, is_avail in available.items() if is_avail]
    
    if len(available_models) == 0:
        return False
    
    return f"{len(available_models)} models available: {', '.join(available_models)}"


def test_create_synthetic_data(runner: TestRunner):
    """Test 4: Create synthetic time series data"""
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252)  # 1 year
    
    # Realistic financial returns
    trend = np.linspace(1.0, 1.2, 252)
    noise = np.random.randn(252) * 0.02
    returns = trend + noise
    data = pd.Series(returns, index=dates)
    
    return f"Created {len(data)} data points (mean: {data.mean():.4f}, std: {data.std():.4f})"


def test_model_prediction(model_name: str, data: pd.Series, runner: TestRunner):
    """Test N: Predict with a specific model"""
    
    from ml.timeseries_predictors import TimeSeriesPredictor
    
    # Check if model is available
    available = TimeSeriesPredictor.list_available_models()
    if not available.get(model_name, False):
        runner.warn(f"{model_name} not available (skipping)")
        return True  # Don't fail, just skip
    
    # Initialize predictor
    predictor = TimeSeriesPredictor(model=model_name)
    
    # Fit
    predictor.fit(data)
    
    # Predict
    result = predictor.predict(horizon=30)
    
    # Validate result
    assert 'predictions' in result, "Missing 'predictions' in result"
    assert len(result['predictions']) == 30, f"Expected 30 predictions, got {len(result['predictions'])}"
    assert all(isinstance(x, (int, float)) for x in result['predictions']), "Invalid prediction type"
    
    mean_pred = np.mean(result['predictions'])
    return f"Predicted 30 days (mean: {mean_pred:.4f})"


def test_model_validation(model_name: str, data: pd.Series, runner: TestRunner):
    """Test N: Validate model on test data"""
    
    from ml.timeseries_predictors import TimeSeriesPredictor
    
    # Check if model is available
    available = TimeSeriesPredictor.list_available_models()
    if not available.get(model_name, False):
        return True  # Skip if not available
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Validate
    predictor = TimeSeriesPredictor(model=model_name)
    metrics = predictor.validate(train_data, test_data)
    
    # Check metrics
    assert 'rmse' in metrics, "Missing RMSE"
    assert 'mae' in metrics, "Missing MAE"
    assert metrics['rmse'] > 0, "Invalid RMSE"
    
    return f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}"


def test_compare_models(data: pd.Series, runner: TestRunner):
    """Test: Compare multiple models"""
    
    from ml.timeseries_predictors import compare_all_models
    
    comparison = compare_all_models(data, test_size=0.2)
    
    assert len(comparison) > 0, "No models compared"
    assert 'model' in comparison.columns, "Missing 'model' column"
    assert 'rmse' in comparison.columns, "Missing 'rmse' column"
    
    best_model = comparison.iloc[0]['model']
    best_rmse = comparison.iloc[0]['rmse']
    
    return f"Compared {len(comparison)} models. Best: {best_model} (RMSE: {best_rmse:.4f})"


def test_get_best_model(data: pd.Series, runner: TestRunner):
    """Test: Get best model automatically"""
    
    from ml.timeseries_predictors import get_best_model
    
    best = get_best_model(data, test_size=0.2, metric='rmse')
    
    assert isinstance(best, str), "Best model is not a string"
    
    return f"Best model: {best}"


def test_ensemble(data: pd.Series, runner: TestRunner):
    """Test: Enhanced Ensemble predictor"""
    
    from ml.timeseries_predictors import TimeSeriesPredictor
    
    # Check if ensemble is available
    available = TimeSeriesPredictor.list_available_models()
    if not available.get('ensemble', False):
        return False
    
    # Test ensemble
    predictor = TimeSeriesPredictor(model='ensemble', auto_weight=True)
    predictor.fit(data)
    
    result = predictor.predict(horizon=30)
    
    # Check ensemble properties
    num_models = len(predictor.predictor.predictors)
    weights = predictor.predictor.weights
    
    return f"{num_models} models in ensemble. Weights: {weights}"


def test_quick_predict(data: pd.Series, runner: TestRunner):
    """Test: Quick predict helper"""
    
    from ml.timeseries_predictors import quick_predict
    
    result = quick_predict(data, horizon=30, model='ensemble')
    
    assert 'predictions' in result
    assert len(result['predictions']) == 30
    
    return "Quick predict works"


def test_feature_engineering(data: pd.Series, runner: TestRunner):
    """Test: Feature engineering for tree models"""
    
    from ml.timeseries_predictors import FeatureEngineer
    
    fe = FeatureEngineer()
    df = fe.create_features(data, lookback=20)
    
    assert len(df) > 0, "No features created"
    assert 'target' in df.columns, "Missing target column"
    assert 'lag_1' in df.columns, "Missing lag features"
    
    num_features = len(df.columns) - 1  # Exclude target
    
    return f"Created {num_features} features from {len(df)} samples"


# =============================================================================
# MAIN TEST SUITE
# =============================================================================

def run_all_tests():
    """Run complete test suite"""
    
    print("="*80)
    print(f"{Fore.CYAN}{Style.BRIGHT}PyManager ML Module v2.0 - Installation Test{Style.RESET_ALL}")
    print("="*80)
    
    runner = TestRunner()
    
    # Phase 1: Basic imports
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"PHASE 1: Library Checks")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    runner.test("Import Libraries", test_imports, runner)
    runner.test("Import ML Module", test_ml_module_import, runner)
    runner.test("List Available Models", test_list_models, runner)
    
    # Phase 2: Data preparation
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"PHASE 2: Data Preparation")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    runner.test("Create Synthetic Data", test_create_synthetic_data, runner)
    runner.test("Feature Engineering", test_feature_engineering, 
                pd.Series(np.random.randn(252)), runner)
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252)
    trend = np.linspace(1.0, 1.2, 252)
    noise = np.random.randn(252) * 0.02
    data = pd.Series(trend + noise, index=dates)
    
    # Phase 3: Individual model tests
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"PHASE 3: Individual Model Tests")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    models_to_test = ['xgboost', 'lightgbm', 'catboost', 'prophet', 'arima']
    
    for model in models_to_test:
        runner.test(f"Predict with {model.upper()}", 
                   test_model_prediction, model, data, runner)
    
    # Phase 4: Validation tests
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"PHASE 4: Model Validation")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    for model in ['xgboost', 'prophet']:  # Test 2 models for speed
        runner.test(f"Validate {model.upper()}", 
                   test_model_validation, model, data, runner)
    
    # Phase 5: Advanced features
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"PHASE 5: Advanced Features")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    runner.test("Compare All Models", test_compare_models, data, runner)
    runner.test("Get Best Model", test_get_best_model, data, runner)
    runner.test("Enhanced Ensemble", test_ensemble, data, runner)
    runner.test("Quick Predict", test_quick_predict, data, runner)
    
    # Summary
    return runner.summary()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n{Fore.RED}FATAL ERROR: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
