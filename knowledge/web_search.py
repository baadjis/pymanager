# knowledge/web_search.py
"""
Web Search Module - Multi-sources
Integrates: DuckDuckGo, Wikipedia
No API keys needed, all free!
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

# DuckDuckGo
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logging.warning("duckduckgo-search not installed. Install: pip install duckduckgo-search")

# Wikipedia
try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False
    logging.warning("wikipedia-api not installed. Install: pip install wikipedia-api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchEngine:
    """
    Multi-source web search engine
    Sources: DuckDuckGo, Wikipedia
    All free, no API keys needed!
    """
    
    def __init__(self, cache_dir: str = "knowledge/cache", cache_ttl: int = 86400):
        """
        Initialize search engine
        
        Args:
            cache_dir: Directory for caching results
            cache_ttl: Cache time-to-live in seconds (default: 24h)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        
        # Initialize Wikipedia
        if WIKI_AVAILABLE:
            self.wiki_fr = wikipediaapi.Wikipedia(
                language='fr',
                user_agent='PyManager/2.0 (Educational Portfolio Management; contact@pymanager.dev)'
            )
            self.wiki_en = wikipediaapi.Wikipedia(
                language='en',
                user_agent='PyManager/2.0 (Educational Portfolio Management; contact@pymanager.dev)'
            )
        else:
            self.wiki_fr = None
            self.wiki_en = None
        
        logger.info("WebSearchEngine initialized")
    
    def search(self, query: str, sources: List[str] = None, max_results: int = 5) -> Dict:
        """
        Search across multiple sources
        
        Args:
            query: Search query
            sources: List of sources to search ['duckduckgo', 'wikipedia', 'all']
            max_results: Maximum results per source
            
        Returns:
            Dictionary with results from each source
        """
        if sources is None:
            sources = ['all']
        
        if 'all' in sources:
            sources = ['duckduckgo', 'wikipedia']
        
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Check cache first
        cached = self._get_from_cache(query)
        if cached:
            logger.info(f"✓ Retrieved from cache: {query}")
            return cached
        
        # DuckDuckGo
        if 'duckduckgo' in sources and DDGS_AVAILABLE:
            try:
                logger.info(f"🔍 Searching DuckDuckGo: {query}")
                results['sources']['duckduckgo'] = self._search_duckduckgo(query, max_results)
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")
                results['sources']['duckduckgo'] = {'error': str(e), 'results': []}
        
        # Wikipedia
        if 'wikipedia' in sources and WIKI_AVAILABLE:
            try:
                logger.info(f"📚 Searching Wikipedia: {query}")
                results['sources']['wikipedia'] = self._search_wikipedia(query)
            except Exception as e:
                logger.error(f"Wikipedia search failed: {e}")
                results['sources']['wikipedia'] = {'error': str(e), 'found': False}
        
        # Cache results
        self._save_to_cache(query, results)
        
        return results
    
    def _search_duckduckgo(self, query: str, max_results: int = 5) -> Dict:
        """
        Search using DuckDuckGo (Free, no API key needed)
        """
        if not DDGS_AVAILABLE:
            return {'error': 'DuckDuckGo not available. Install: pip install duckduckgo-search', 'results': []}
        
        try:
            # Add finance context to query
            finance_query = f"{query} finance investment"
            
            ddgs = DDGS()
            
            # Text search
            text_results = list(ddgs.text(finance_query, max_results=max_results))
            
            # Format results
            formatted_results = []
            for result in text_results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', ''),
                    'source': 'duckduckgo'
                })
            
            logger.info(f"✓ DuckDuckGo: {len(formatted_results)} results")
            
            return {
                'count': len(formatted_results),
                'results': formatted_results
            }
        
        except Exception as e:
            logger.error(f"DuckDuckGo error: {e}")
            return {'error': str(e), 'results': []}
    
    def _search_wikipedia(self, query: str) -> Dict:
        """
        Search Wikipedia for financial topics
        """
        if not WIKI_AVAILABLE:
            return {
                'error': 'Wikipedia not available. Install: pip install wikipedia-api',
                'found': False
            }
        
        try:
            # Try common variations in French first
            search_terms = [
                query,
                query.replace('ratio de ', ''),
                query.replace('modèle de ', ''),
                query.replace('théorie de ', ''),
                query.title(),
                query.capitalize()
            ]
            
            # Try French Wikipedia first
            for term in search_terms:
                page = self.wiki_fr.page(term)
                if page.exists():
                    logger.info(f"✓ Wikipedia FR: Found '{page.title}'")
                    # Get section titles safely
                    section_titles = []
                    try:
                        if hasattr(page, 'sections') and page.sections:
                            if isinstance(page.sections, dict):
                                section_titles = list(page.sections.keys())[:5]
                            elif isinstance(page.sections, list):
                                section_titles = [s.title for s in page.sections[:5] if hasattr(s, 'title')]
                    except:
                        section_titles = []
                    
                    return {
                        'found': True,
                        'title': page.title,
                        'summary': page.summary[:500] + '...' if len(page.summary) > 500 else page.summary,
                        'url': page.fullurl,
                        'language': 'fr',
                        'sections': section_titles,
                        'source': 'wikipedia'
                    }
            
            # Try English Wikipedia
            for term in search_terms:
                page = self.wiki_en.page(term)
                if page.exists():
                    logger.info(f"✓ Wikipedia EN: Found '{page.title}'")
                    return {
                        'found': True,
                        'title': page.title,
                        'summary': page.summary[:500] + '...' if len(page.summary) > 500 else page.summary,
                        'url': page.fullurl,
                        'language': 'en',
                        'sections': list(page.sections.keys())[:5] if page.sections else [],
                        'source': 'wikipedia'
                    }
            
            # Not found
            logger.info(f"✗ Wikipedia: No article found for '{query}'")
            return {
                'found': False,
                'message': f"No Wikipedia article found for: {query}"
            }
        
        except Exception as e:
            logger.error(f"Wikipedia error: {e}")
            return {'error': str(e), 'found': False}
    
    def get_wikipedia_article(self, title: str, language: str = 'fr', section: Optional[str] = None) -> Dict:
        """
        Get full Wikipedia article or specific section
        
        Args:
            title: Article title
            language: 'fr' or 'en'
            section: Optional section name
            
        Returns:
            Article content
        """
        if not WIKI_AVAILABLE:
            return {'error': 'Wikipedia not available'}
        
        try:
            wiki = self.wiki_fr if language == 'fr' else self.wiki_en
            page = wiki.page(title)
            
            if not page.exists():
                return {'error': f"Article '{title}' not found"}
            
            result = {
                'title': page.title,
                'url': page.fullurl,
                'summary': page.summary,
                'language': language
            }
            
            if section and section in page.sections:
                section_obj = page.sections[section]
                result['section'] = {
                    'title': section,
                    'text': section_obj.text
                }
            else:
                result['full_text'] = page.text[:5000]  # Limit to 5000 chars
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting Wikipedia article: {e}")
            return {'error': str(e)}
    
    def search_financial_term(self, term: str) -> str:
        """
        Search for a financial term across all sources and synthesize
        
        Args:
            term: Financial term to search
            
        Returns:
            Synthesized information from multiple sources
        """
        results = self.search(term, sources=['all'], max_results=3)
        
        # Build synthesized response
        response = f"## 🔍 Résultats pour: {term}\n\n"
        
        # Wikipedia (most authoritative)
        if 'wikipedia' in results['sources']:
            wiki = results['sources']['wikipedia']
            if wiki.get('found'):
                response += f"### 📚 Wikipedia ({wiki.get('language', 'fr').upper()})\n\n"
                response += f"**{wiki['title']}**\n\n"
                response += f"{wiki['summary']}\n\n"
                response += f"[Lire plus]({wiki['url']})\n\n"
                
                if wiki.get('sections'):
                    response += f"**Sections disponibles:** {', '.join(wiki['sections'][:3])}\n\n"
        
        # DuckDuckGo results
        if 'duckduckgo' in results['sources']:
            ddg = results['sources']['duckduckgo']
            if ddg.get('results'):
                response += f"### 🌐 Sources Web\n\n"
                for i, result in enumerate(ddg['results'][:3], 1):
                    response += f"{i}. **{result['title']}**\n"
                    response += f"   {result['snippet']}\n"
                    response += f"   [Source]({result['url']})\n\n"
        
        if len(response) < 100:  # No meaningful results
            response = f"❌ Aucun résultat trouvé pour: {term}\n\n"
            response += "**Suggestions:**\n"
            response += "- Vérifiez l'orthographe\n"
            response += "- Utilisez des termes plus généraux\n"
            response += "- Essayez en anglais\n\n"
            response += "💡 Exemple: 'Sharpe ratio' au lieu de 'ratio de sharpe'\n"
        
        return response
    
    def _get_cache_path(self, query: str) -> Path:
        """Get cache file path for a query"""
        # Create safe filename
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_'))
        safe_query = safe_query.replace(' ', '_')[:50]
        return self.cache_dir / f"{safe_query}.json"
    
    def _get_from_cache(self, query: str) -> Optional[Dict]:
        """Get results from cache if not expired"""
        cache_file = self._get_cache_path(query)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check if expired
            cached_time = datetime.fromisoformat(cached['timestamp'])
            age = datetime.now() - cached_time
            
            if age > timedelta(seconds=self.cache_ttl):
                logger.info(f"Cache expired for: {query} (age: {age})")
                cache_file.unlink()  # Delete expired cache
                return None
            
            logger.info(f"Cache hit for: {query} (age: {age})")
            return cached
        
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def _save_to_cache(self, query: str, results: Dict):
        """Save results to cache"""
        cache_file = self._get_cache_path(query)
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cached results for: {query}")
        
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def clear_cache(self):
        """Clear all cached results"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        logger.info(f"Cleared {count} cache files")
        return count
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.json"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # Count expired
        expired = 0
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                cached_time = datetime.fromisoformat(cached['timestamp'])
                age = datetime.now() - cached_time
                if age > timedelta(seconds=self.cache_ttl):
                    expired += 1
            except:
                pass
        
        return {
            'total_files': len(cache_files),
            'total_size_kb': total_size / 1024,
            'expired_files': expired,
            'cache_dir': str(self.cache_dir)
        }


# =============================================================================
# Helper Functions
# =============================================================================

def search_financial_topics(topics: List[str]) -> Dict[str, str]:
    """
    Search multiple financial topics
    
    Args:
        topics: List of topics to search
        
    Returns:
        Dictionary mapping topics to synthesized results
    """
    search_engine = WebSearchEngine()
    results = {}
    
    for topic in topics:
        logger.info(f"Searching: {topic}")
        results[topic] = search_engine.search_financial_term(topic)
    
    return results


def quick_search(query: str) -> str:
    """Quick search function"""
    search = WebSearchEngine()
    return search.search_financial_term(query)


# =============================================================================
# Test & Main
# =============================================================================

if __name__ == "__main__":
    # Test Web Search
    print("🔍 Testing Web Search Engine\n")
    print("="*60)
    
    search_engine = WebSearchEngine()
    
    # Test queries
    test_queries = [
        "ratio de Sharpe",
        "diversification portfolio",
        "Black-Litterman model"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = search_engine.search_financial_term(query)
        print(result)
        
        # Pause between requests
        import time
        time.sleep(1)
    
    # Cache stats
    print(f"\n{'='*60}")
    print("📊 Cache Statistics")
    print('='*60)
    
    stats = search_engine.get_cache_stats()
    print(f"Total cache files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_kb']:.1f} KB")
    print(f"Expired files: {stats['expired_files']}")
    print(f"Cache directory: {stats['cache_dir']}")
    
    print("\n✅ Web Search test complete!")
    print("\n💡 Install dependencies:")
    print("   pip install duckduckgo-search wikipedia-api")
