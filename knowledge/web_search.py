# knowledge/web_search.py
"""
Optimized Web Search - Finance Oriented
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logging.warning("duckduckgo-search not installed")

try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False
    logging.warning("wikipedia-api not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FINANCE OPTIMIZATIONS
# =============================================================================

# Map French → English financial terms
FINANCE_TRANSLATIONS = {
    'value at risk': 'value at risk var',
    'var': 'value at risk var',
    'valeur à risque': 'value at risk',
    'ratio de sharpe': 'sharpe ratio',
    'sharpe': 'sharpe ratio',
    'ratio de sortino': 'sortino ratio',
    'sortino': 'sortino ratio',
    'markowitz': 'markowitz modern portfolio theory',
    'black-litterman': 'black litterman model',
    'capm': 'capital asset pricing model capm',
    'diversification': 'portfolio diversification',
    'volatilité': 'volatility finance',
    'rendement': 'return investment',
    'portefeuille': 'portfolio investment',
    'allocation': 'asset allocation',
    'optimisation': 'portfolio optimization',
    'efficient frontier': 'efficient frontier markowitz',
    'frontière efficiente': 'efficient frontier markowitz',
}

# Priority sources for financial terms
FINANCE_SOURCES = [
    'investopedia',
    'wikipedia',
    'cfa institute',
    'finance',
    'investment',
    'portfolio',
    'financialpipeline'
]

class WebSearchEngine:
    """Finance-optimized search engine"""
    
    def __init__(self, cache_dir: str = "knowledge/cache", cache_ttl: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        
        if WIKI_AVAILABLE:
            self.wiki_fr = wikipediaapi.Wikipedia(
                language='fr',
                user_agent='PyManager/2.0 (Educational; finance@pymanager.dev)'
            )
            self.wiki_en = wikipediaapi.Wikipedia(
                language='en',
                user_agent='PyManager/2.0 (Educational; finance@pymanager.dev)'
            )
        else:
            self.wiki_fr = None
            self.wiki_en = None
        
        logger.info("WebSearchEngine initialized (finance-optimized)")
    
    def search(self, query: str, sources: List[str] = None, max_results: int = 5) -> Dict:
        """
        Search with finance optimization
        
        Auto-translates French → English
        Prioritizes financial sources
        Filters non-finance results
        """
        if sources is None:
            sources = ['all']
        
        if 'all' in sources:
            sources = ['wikipedia', 'duckduckgo']
        
        # Optimize query for finance
        optimized_query = self._optimize_query(query)
        
        logger.info(f"Original: '{query}' → Optimized: '{optimized_query}'")
        
        # Check cache
        cached = self._get_from_cache(optimized_query)
        if cached:
            logger.info("✓ From cache")
            return cached
        
        results = {
            'query': query,
            'optimized_query': optimized_query,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Wikipedia (try EN first for finance terms)
        if 'wikipedia' in sources and WIKI_AVAILABLE:
            logger.info("📚 Searching Wikipedia (EN priority)...")
            results['sources']['wikipedia'] = self._search_wikipedia_optimized(optimized_query)
           
        
        # DuckDuckGo with finance filter
        if 'duckduckgo' in sources and DDGS_AVAILABLE:
            logger.info("🦆 Searching DuckDuckGo (finance filtered)...")
            results['sources']['duckduckgo'] = self._search_duckduckgo_optimized(optimized_query, max_results)
        
        #self._save_to_cache(optimized_query, results)
        
        return results
    
    def _optimize_query(self, query: str) -> str:
        """
        Optimize query for finance search
        
        1. Translate FR → EN
        2. Add finance context
        3. Expand acronyms
        """
        query_lower = query.lower().strip()
        
        # Direct translation match
        for fr_term, en_term in FINANCE_TRANSLATIONS.items():
            if fr_term in query_lower:
                logger.info(f"  Translated: {fr_term} → {en_term}")
                return en_term
        
        # Extract main concept and add finance context
        # Remove question words
        clean = query_lower
        for word in ['explique', 'explique-moi', 'qu\'est-ce que', 'comment', 'définition', 'parle-moi']:
            clean = clean.replace(word, '')
        
        clean = clean.replace('la ', '').replace('le ', '').replace('les ', '').replace('l\'', '')
        clean = clean.strip().rstrip('?').strip()
        
        # If not in translations, add finance context
        if clean and not any(term in clean for term in ['finance', 'investment', 'portfolio']):
            return f"{clean} finance investment"
        
        return clean or query
    
    def _search_wikipedia_optimized(self, query: str) -> Dict:
        """
        Search Wikipedia with finance optimization
        
        Priority: EN > FR (finance content better in English)
        """
        if not WIKI_AVAILABLE:
            return {'found': False, 'error': 'Wikipedia not available'}
        
        # Generate search variations
        variations = self._generate_variations(query)
        
        # Try English first (better finance content)
        for term in variations:
            page = self.wiki_en.page(term)
            if page.exists():
                logger.info(f"  ✓ Found (EN): {page.title}")
                return self._format_wiki_result(page, 'en')
        
        # Try French as fallback
        for term in variations:
            page = self.wiki_fr.page(term)
            if page.exists():
                logger.info(f"  ✓ Found (FR): {page.title}")
                return self._format_wiki_result(page, 'fr')
        
        logger.info(f"  ✗ Not found: {query}")
        return {'found': False, 'message': f'No article for: {query}'}
    
    def _generate_variations(self, query: str) -> List[str]:
        """Generate search variations for better matching"""
        variations = [
            query,
            query.title(),
            query.capitalize(),
            query.replace('-', ' '),
            query.replace(' ', '-'),
        ]
        
        # Add with "ratio", "model", etc. removed
        for term in [' ratio', ' model', ' theory', ' method']:
            if term in query.lower():
                variations.append(query.lower().replace(term, ''))
        
        # Deduplicate
        return list(dict.fromkeys(variations))
    
    def _format_wiki_result(self, page, lang: str) -> Dict:
        """Format Wikipedia result"""
        # Get sections safely
        sections = []
        try:
            if hasattr(page, 'sections'):
                if isinstance(page.sections, list):
                    sections = [s.title for s in page.sections[:5] if hasattr(s, 'title')]
        except:
            pass
        
        return {
            'found': True,
            'title': page.title,
            'summary': page.summary[:800] + '...' if len(page.summary) > 800 else page.summary,
            'url': page.fullurl,
            'language': lang,
            'sections': sections,
            'source': 'wikipedia'
        }
    
    def _search_duckduckgo_optimized(self, query: str, max_results: int = 5) -> Dict:
        """
        Search DuckDuckGo with finance filtering
        
        Prioritizes:
        - Investopedia
        - Wikipedia
        - .edu sites
        - Finance publications
        """
        if not DDGS_AVAILABLE:
            return {'error': 'DuckDuckGo not available', 'results': []}
        
        try:
            import time
            time.sleep(0.5)  # Rate limiting
            
            ddgs = DDGS()
            
            # Search with more results to filter
            raw_results = list(ddgs.text(query, max_results=20))
            
            # Filter and score results
            scored_results = []
            for result in raw_results:
                score = self._score_finance_relevance(result)
                if score > 0:
                    scored_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('body', ''),
                        'url': result.get('href', ''),
                        'source': 'duckduckgo',
                        'relevance_score': score
                    })
            
            # Sort by relevance and take top N
            scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            final_results = scored_results[:max_results]
            
            logger.info(f"  ✓ {len(final_results)}/{len(raw_results)} finance-relevant results")
            
            return {
                'count': len(final_results),
                'results': final_results
            }
        
        except Exception as e:
            logger.error(f"DuckDuckGo error: {e}")
            return {'error': str(e), 'results': []}
    
    def _score_finance_relevance(self, result: Dict) -> float:
        """
        Score result relevance for finance
        
        Higher score = more relevant
        0 = not relevant (filtered out)
        """
        url = result.get('href', '').lower()
        title = result.get('title', '').lower()
        snippet = result.get('body', '').lower()
        
        score = 0.0
        
        # High priority sources (+3.0)
        if 'investopedia.com' in url or 'financedemarche.fr' in url:
            score += 3.0
        elif 'wikipedia.org' in url:
            score += 2.5
        
        elif any(domain in url for domain in ['.edu', 'cfa', 'cfainstitute','financialpipeline']):
            score += 2.0
        elif any(domain in url for domain in ['bloomberg', 'reuters', 'financial', 'wsj']):
            score += 1.5
        
        # Finance keywords (+0.5 each)
        finance_keywords = [
            'finance', 'investment', 'portfolio', 'risk', 'return',
            'volatility', 'sharpe', 'ratio', 'asset', 'market',
            'trading', 'stock', 'bond', 'equity', 'derivative'
        ]
        
        text = f"{title} {snippet}"
        for keyword in finance_keywords:
            if keyword in text:
                score += 0.3
        
        # Negative signals
        if any(word in url for word in ['shopping', 'amazon', 'ebay', 'reddit']):
            return 0.0
        
        if any(word in title for word in ['buy', 'sale', 'discount', 'price']):
            score -= 1.0
        
        return max(0.0, score)
    
    def search_financial_term(self, term: str) -> str:
        """
        Search and synthesize for financial term
        Optimized output format
        """
        results = self.search(term, sources=['all'], max_results=3)
        
        # Build clean response
        response = f"## 📊 {results['query']}\n\n"
        
        if results.get('optimized_query') != results['query']:
            response += f"*Recherche optimisée: {results['optimized_query']}*\n\n"
        
        # Wikipedia (primary source)
        wiki = results['sources'].get('wikipedia', {})
        if wiki.get('found'):
            response += f"### 📖 {wiki['title']}\n\n"
            response += f"{wiki['summary']}\n\n"
            response += f"[Wikipedia {wiki['language'].upper()}]({wiki['url']})\n\n"
        
        # Web sources (filtered)
        ddg = results['sources'].get('duckduckgo', {})
        if ddg.get('results'):
            response += f"### 🌐 Sources spécialisées\n\n"
            for i, r in enumerate(ddg['results'][:3], 1):
                score_icon = "⭐" * min(3, int(r.get('relevance_score', 0)))
                response += f"{i}. {score_icon} **{r['title']}**\n"
                response += f"   {r['snippet'][:150]}...\n"
                response += f"   [{self._extract_domain(r['url'])}]({r['url']})\n\n"
        
        # No results
        if not wiki.get('found') and not ddg.get('results'):
            response = f"❌ Aucun résultat pour: {term}\n\n"
            response += "**Essayez:**\n"
            response += "- Termes en anglais (ex: 'Value at Risk')\n"
            response += "- Acronymes (VaR, CAPM, etc.)\n"
            response += "- Termes plus généraux\n"
        
        return response
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain name"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain.replace('www.', '')
        except:
            return 'source'
    
    # Cache methods (unchanged)
    def _get_cache_path(self, query: str) -> Path:
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_'))
        safe_query = safe_query.replace(' ', '_')[:50]
        return self.cache_dir / f"{safe_query}.json"
    
    def _get_from_cache(self, query: str) -> Optional[Dict]:
        cache_file = self._get_cache_path(query)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            cached_time = datetime.fromisoformat(cached['timestamp'])
            age = datetime.now() - cached_time
            if age > timedelta(seconds=self.cache_ttl):
                cache_file.unlink()
                return None
            return cached
        except:
            return None
    
    def _save_to_cache(self, query: str, results: Dict):
        cache_file = self._get_cache_path(query)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    def clear_cache(self):
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
    
    def get_cache_stats(self) -> Dict:
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            'total_files': len(files),
            'total_size_kb': total_size / 1024,
            'cache_dir': str(self.cache_dir)
        }

# Quick search helper
def quick_search(query: str) -> str:
    search = WebSearchEngine()
    return search.search_financial_term(query)

# Test
if __name__ == "__main__":
    print("🔍 Testing Finance-Optimized Search\n")
    print("="*70)
    
    search = WebSearchEngine()
    
    test_queries = [
        " the Value at Risk (V.A.R)",
        "ratio de sharpe", 
        "diversification portefeuille"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        result = search.search_financial_term(query)
        print(result)
        
        import time
        time.sleep(2)
    
    print(f"\n{'='*70}")
    print("✅ Test complete")
