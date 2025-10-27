# knowledge/fed_data.py
"""
FRED Economic Data Module
Federal Reserve Economic Data integration
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("fredapi not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FREDDataProvider:
    """
    Provides economic data from FRED (Federal Reserve Economic Data)
    Useful for macroeconomic context in investment decisions
    """
    
    # Common economic indicators
    INDICATORS = {
        # Interest Rates
        'DFF': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Rate',
        'DGS2': '2-Year Treasury Rate',
        'T10Y2Y': '10-Year minus 2-Year Treasury Spread',
        
        # Inflation
        'CPIAUCSL': 'Consumer Price Index (CPI)',
        'CPILFESL': 'Core CPI (excluding food & energy)',
        'PCEPI': 'Personal Consumption Expenditures Price Index',
        
        # GDP & Growth
        'GDP': 'Gross Domestic Product',
        'GDPC1': 'Real GDP',
        'A191RL1Q225SBEA': 'Real GDP Growth Rate',
        
        # Employment
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'CIVPART': 'Labor Force Participation Rate',
        
        # Markets
        'SP500': 'S&P 500 Index',
        'VIXCLS': 'VIX (Volatility Index)',
        'DEXUSEU': 'USD/EUR Exchange Rate',
        
        # Monetary
        'M2SL': 'M2 Money Supply',
        'WALCL': 'Fed Balance Sheet',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data provider
        
        Args:
            api_key: FRED API key (get free at https://fred.stlouisfed.org/)
        """
        if not FRED_AVAILABLE:
            logger.error("fredapi not installed. Install with: pip install fredapi")
            self.fred = None
            return
        
        if not api_key:
            logger.warning("No FRED API key provided. Limited functionality.")
            self.fred = None
            return
        
        try:
            self.fred = Fred(api_key=api_key)
            logger.info("FRED API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FRED API: {e}")
            self.fred = None
    
    def get_indicator(
        self, 
        series_id: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        Get economic indicator data
        
        Args:
            series_id: FRED series ID (e.g., 'DFF', 'UNRATE')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Pandas Series with the data
        """
        if not self.fred:
            return None
        
        try:
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            
            logger.info(f"Retrieved {len(data)} observations for {series_id}")
            return data
        
        except Exception as e:
            logger.error(f"Error getting {series_id}: {e}")
            return None
    
    def get_latest_value(self, series_id: str) -> Optional[float]:
        """
        Get latest value for an indicator
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Latest value or None
        """
        data = self.get_indicator(series_id)
        
        if data is not None and len(data) > 0:
            return float(data.iloc[-1])
        
        return None
    
    def get_economic_summary(self) -> Dict:
        """
        Get summary of key economic indicators
        
        Returns:
            Dictionary with current values and trends
        """
        if not self.fred:
            return {'error': 'FRED API not available'}
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'indicators': {}
        }
        
        # Key indicators to fetch
        key_indicators = {
            'DFF': 'Fed Funds Rate',
            'DGS10': '10Y Treasury',
            'UNRATE': 'Unemployment',
            'CPIAUCSL': 'CPI',
            'SP500': 'S&P 500',
            'VIXCLS': 'VIX'
        }
        
        for series_id, name in key_indicators.items():
            try:
                latest = self.get_latest_value(series_id)
                if latest is not None:
                    summary['indicators'][name] = {
                        'value': latest,
                        'series_id': series_id
                    }
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
        
        return summary
    
    def get_recession_indicators(self) -> Dict:
        """
        Get recession indicators (yield curve, unemployment, etc.)
        
        Returns:
            Dictionary with recession signals
        """
        if not self.fred:
            return {'error': 'FRED API not available'}
        
        indicators = {}
        
        # 1. Yield Curve (10Y - 2Y)
        try:
            yield_spread = self.get_latest_value('T10Y2Y')
            if yield_spread is not None:
                indicators['yield_curve'] = {
                    'spread': yield_spread,
                    'inverted': yield_spread < 0,
                    'signal': 'Warning' if yield_spread < 0 else 'Normal',
                    'description': 'Inverted yield curve often precedes recession'
                }
        except:
            pass
        
        # 2. Unemployment Rate
        try:
            unemployment = self.get_latest_value('UNRATE')
            if unemployment is not None:
                indicators['unemployment'] = {
                    'rate': unemployment,
                    'signal': 'High' if unemployment > 6 else 'Normal',
                    'description': f'Current unemployment: {unemployment}%'
                }
        except:
            pass
        
        # 3. VIX (Fear Index)
        try:
            vix = self.get_latest_value('VIXCLS')
            if vix is not None:
                indicators['vix'] = {
                    'value': vix,
                    'signal': 'High Fear' if vix > 30 else 'Elevated' if vix > 20 else 'Normal',
                    'description': f'VIX at {vix:.1f} (>30 = high fear)'
                }
        except:
            pass
        
        return {
            'timestamp': datetime.now().isoformat(),
            'indicators': indicators,
            'overall_signal': self._assess_recession_risk(indicators)
        }
    
    def _assess_recession_risk(self, indicators: Dict) -> str:
        """Assess overall recession risk from indicators"""
        warnings = 0
        total = len(indicators)
        
        if 'yield_curve' in indicators and indicators['yield_curve'].get('inverted'):
            warnings += 1
        
        if 'unemployment' in indicators and indicators['unemployment']['signal'] == 'High':
            warnings += 1
        
        if 'vix' in indicators and indicators['vix']['signal'] == 'High Fear':
            warnings += 1
        
        if total == 0:
            return 'Unknown'
        
        risk_pct = (warnings / total) * 100
        
        if risk_pct >= 66:
            return 'High Risk'
        elif risk_pct >= 33:
            return 'Moderate Risk'
        else:
            return 'Low Risk'
    
    def get_inflation_data(self, months: int = 12) -> Dict:
        """
        Get recent inflation data
        
        Args:
            months: Number of months of data
            
        Returns:
            Inflation metrics
        """
        if not self.fred:
            return {'error': 'FRED API not available'}
        
        start_date = (datetime.now() - timedelta(days=months*30)).strftime('%Y-%m-%d')
        
        try:
            cpi = self.get_indicator('CPIAUCSL', start_date=start_date)
            
            if cpi is None or len(cpi) < 12:
                return {'error': 'Insufficient data'}
            
            # Calculate YoY inflation
            latest_cpi = cpi.iloc[-1]
            year_ago_cpi = cpi.iloc[-13] if len(cpi) >= 13 else cpi.iloc[0]
            
            yoy_inflation = ((latest_cpi - year_ago_cpi) / year_ago_cpi) * 100
            
            # Recent trend (last 3 months)
            recent_trend = cpi.iloc[-3:].pct_change().mean() * 100
            
            return {
                'latest_cpi': float(latest_cpi),
                'yoy_inflation': float(yoy_inflation),
                'recent_trend': 'Rising' if recent_trend > 0 else 'Falling',
                'target': 2.0,  # Fed's inflation target
                'above_target': yoy_inflation > 2.0,
                'data_points': len(cpi)
            }
        
        except Exception as e:
            logger.error(f"Error calculating inflation: {e}")
            return {'error': str(e)}
    
    def format_economic_context(self) -> str:
        """
        Format economic context for AI assistant
        
        Returns:
            Formatted string with economic context
        """
        if not self.fred:
            return "⚠️ FRED API non disponible. Données économiques limitées."
        
        try:
            summary = self.get_economic_summary()
            recession = self.get_recession_indicators()
            inflation = self.get_inflation_data()
            
            context = "📊 **Contexte Économique Actuel**\n\n"
            
            # Key indicators
            if 'indicators' in summary:
                context += "**Indicateurs Clés:**\n"
                for name, data in summary['indicators'].items():
                    context += f"- {name}: {data['value']:.2f}\n"
                context += "\n"
            
            # Inflation
            if 'yoy_inflation' in inflation:
                context += f"**Inflation:** {inflation['yoy_inflation']:.1f}% (YoY)\n"
                context += f"Tendance: {inflation['recent_trend']}\n\n"
            
            # Recession risk
            if 'overall_signal' in recession:
                context += f"**Risque de Récession:** {recession['overall_signal']}\n"
                if 'yield_curve' in recession['indicators']:
                    yc = recession['indicators']['yield_curve']
                    context += f"- Yield Curve: {yc['signal']} (spread: {yc['spread']:.2f}%)\n"
                context += "\n"
            
            context += f"*Mis à jour: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
            
            return context
        
        except Exception as e:
            logger.error(f"Error formatting economic context: {e}")
            return "⚠️ Erreur lors de la récupération des données économiques."
    
    def list_available_indicators(self) -> Dict[str, str]:
        """Get list of available indicators"""
        return self.INDICATORS.copy()


# =============================================================================
# Helper Functions
# =============================================================================

def get_economic_context(api_key: Optional[str] = None) -> str:
    """
    Quick function to get economic context
    
    Args:
        api_key: FRED API key
        
    Returns:
        Formatted economic context
    """
    provider = FREDDataProvider(api_key)
    return provider.format_economic_context()


if __name__ == "__main__":
    # Test FRED Data Provider
    print("📊 Testing FRED Data Provider\n")
    
    import os
    api_key = os.getenv('FRED_API_KEY')
    
    if not api_key:
        print("⚠️ FRED_API_KEY not set. Get one at: https://fred.stlouisfed.org/")
        print("   Set it: export FRED_API_KEY='your-key'\n")
    
    provider = FREDDataProvider(api_key)
    
    if provider.fred:
        print("✅ FRED API initialized\n")
        
        # Test 1: Latest values
        print("1. Testing latest values...")
        fed_rate = provider.get_latest_value('DFF')
        if fed_rate:
            print(f"   Fed Funds Rate: {fed_rate:.2f}%")
        
        # Test 2: Economic summary
        print("\n2. Economic summary...")
        summary = provider.get_economic_summary()
        if 'indicators' in summary:
            for name, data in summary['indicators'].items():
                print(f"   {name}: {data['value']:.2f}")
        
        # Test 3: Recession indicators
        print("\n3. Recession indicators...")
        recession = provider.get_recession_indicators()
        print(f"   Overall Risk: {recession.get('overall_signal', 'Unknown')}")
        
        # Test 4: Inflation
        print("\n4. Inflation data...")
        inflation = provider.get_inflation_data()
        if 'yoy_inflation' in inflation:
            print(f"   YoY Inflation: {inflation['yoy_inflation']:.1f}%")
        
        # Test 5: Full context
        print("\n5. Full economic context:")
        print("   " + "="*50)
        context = provider.format_economic_context()
        for line in context.split('\n'):
            print(f"   {line}")
    
    else:
        print("❌ FRED API not available")
    
    print("\n✅ FRED test complete!")
