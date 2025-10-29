# knowledge/rag_engine.py
"""
Enhanced RAG Engine - Finance Optimized
Improvements:
- Hybrid search (semantic + keyword BM25)
- Automatic document expansion from web
- Financial term extraction
- Query rewriting
- Relevance re-ranking
- No extra infrastructure needed!
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pickle
import logging
import re
from collections import Counter
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRAG:
    """Enhanced RAG with hybrid search and auto-expansion"""
    
    def __init__(self, knowledge_dir: str = "knowledge", model_name: str = "all-MiniLM-L6-v2"):
        self.knowledge_dir = Path(knowledge_dir)
        self.documents_dir = self.knowledge_dir / "documents"
        self.embeddings_dir = self.knowledge_dir / "embeddings"
        
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Storage
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.keywords_index = {}  # NEW: Keyword index for BM25
        self.term_frequencies = {}  # NEW: For BM25 scoring
        
        # Financial terms dictionary (auto-expansion)
        self.finance_terms = self._load_finance_terms()
        
        self.load_index()
        
        # Auto-populate if empty
        if len(self.documents) == 0:
            logger.info("Index empty, auto-populating...")
            self._auto_populate()
    
    def _load_finance_terms(self) -> Dict[str, List[str]]:
        """Financial terms and synonyms for query expansion"""
        return {
            'sharpe': ['sharpe ratio', 'rendement ajusté au risque', 'risk-adjusted return'],
            'var': ['value at risk', 'var', 'valeur à risque', 'perte maximale'],
            'sortino': ['sortino ratio', 'downside risk', 'risque à la baisse'],
            'markowitz': ['markowitz', 'modern portfolio theory', 'mpt', 'frontière efficiente', 'efficient frontier'],
            'diversification': ['diversification', 'répartition', 'allocation', 'spread risk'],
            'volatility': ['volatilité', 'volatility', 'écart-type', 'standard deviation', 'risque'],
            'capm': ['capm', 'capital asset pricing model', 'modèle d\'évaluation des actifs'],
            'black-litterman': ['black-litterman', 'bl model', 'equilibrium'],
            'beta': ['beta', 'bêta', 'sensibilité au marché', 'market sensitivity'],
            'alpha': ['alpha', 'surperformance', 'outperformance', 'excess return'],
        }
    
    def _auto_populate(self):
        """Auto-populate with essential financial concepts"""
        core_docs = [
            {
                "title": "Ratio de Sharpe - Mesure de Performance",
                "content": """Le ratio de Sharpe mesure le rendement ajusté au risque d'un investissement.
                Formule: (Rendement - Taux sans risque) / Volatilité (écart-type).
                Interprétation: >2.0 = Excellent, 1.0-2.0 = Très bon, 0.5-1.0 = Acceptable, <0.5 = Faible.
                Développé par William Sharpe en 1966. Permet de comparer des investissements avec différents niveaux de risque.
                Limite: suppose distribution normale des rendements. Utilisé dans PyManager pour évaluer les portfolios optimisés.""",
                "category": "metrics",
                "keywords": ["sharpe", "ratio", "risque", "rendement", "performance", "volatilité"]
            },
            {
                "title": "Value at Risk (VaR) - Mesure du Risque",
                "content": """La VaR estime la perte maximale probable d'un portfolio sur une période avec un niveau de confiance donné.
                Exemple: VaR 95% = 10K€ signifie 95% de chances que perte ≤ 10K€.
                Trois méthodes: 1) Historique (données passées), 2) Paramétrique (distribution normale), 3) Monte Carlo (simulations).
                Limite: ne dit rien au-delà du seuil. Compléter avec CVaR (Expected Shortfall).
                Dans PyManager: disponible dans Portfolio Details > Analytics.""",
                "category": "risk",
                "keywords": ["var", "value at risk", "perte", "risque", "probabilité"]
            },
            {
                "title": "Ratio de Sortino - Risque Asymétrique",
                "content": """Le ratio de Sortino améliore le Sharpe en ne pénalisant que la volatilité négative (downside).
                Formule: (Rendement - Taux cible) / Écart-type downside.
                Avantage: ne pénalise pas les hausses, seulement les baisses. Plus adapté pour investisseurs averses aux pertes.
                Le Sharpe pénalise toute volatilité, le Sortino uniquement le downside. Meilleur pour stratégies asymétriques.""",
                "category": "metrics",
                "keywords": ["sortino", "downside", "risque", "asymétrique", "baisse"]
            },
            {
                "title": "Théorie Moderne du Portfolio (Markowitz)",
                "content": """Développée par Harry Markowitz (1952), optimise le ratio rendement/risque par diversification.
                Concept: Frontière efficiente = ensemble des portfolios optimaux (max rendement pour risque donné).
                Optimisation: maximiser Sharpe sous contraintes (somme poids = 100%, poids ≥ 0%).
                PyManager propose 4 modes: Sharp (max Sharpe), Risk (min volatilité), Return (max rendement), Unsafe (sans contraintes).
                Limites: suppose rendements normaux, basé sur historique, sensible aux estimations.""",
                "category": "theory",
                "keywords": ["markowitz", "mpt", "frontière", "efficiente", "optimisation", "diversification"]
            },
            {
                "title": "Modèle Black-Litterman",
                "content": """Combine équilibre de marché + vues personnelles pour allocation d'actifs.
                Processus: 1) Partir équilibre marché (cap-weighted), 2) Ajouter vos vues avec confiance, 3) Ajuster poids.
                Avantages vs Markowitz: moins sensible aux estimations, poids plus stables, intègre insights de l'investisseur.
                Développé par Fischer Black et Robert Litterman (1990) chez Goldman Sachs.
                Dans PyManager: Build Portfolio > Black-Litterman avec interface de saisie de vues.""",
                "category": "theory",
                "keywords": ["black-litterman", "vues", "equilibrium", "allocation", "bayesian"]
            },
            {
                "title": "Diversification de Portfolio",
                "content": """Stratégie de réduction du risque par répartition sur plusieurs actifs peu corrélés.
                Principe: "Ne pas mettre tous ses œufs dans le même panier". Réduit le risque spécifique (non systématique).
                Dimensions: Actifs (actions/obligations), Secteurs (tech/santé/énergie), Géographie (US/Europe/Asie), 
                Capitalisation (large/mid/small), Style (growth/value/dividend).
                Corrélation clé: <0.3 = bien diversifié, >0.7 = mal diversifié.
                Dans PyManager: vérifier Sector Allocation dans Portfolio Details.""",
                "category": "strategy",
                "keywords": ["diversification", "corrélation", "allocation", "secteurs", "risque"]
            },
            {
                "title": "CAPM - Capital Asset Pricing Model",
                "content": """Modèle d'évaluation des actifs financiers basé sur le risque systématique (bêta).
                Formule: Rendement attendu = Taux sans risque + β × (Rendement marché - Taux sans risque).
                Beta: β=1 (comme marché), β>1 (plus volatil), β<1 (moins volatil).
                Exemple: AAPL β=1.2, Rf=2%, Rm=10% → Rendement attendu = 2% + 1.2×8% = 11.6%.
                Limites: suppose marchés efficients, un seul facteur. Alternative: Fama-French (3 facteurs).""",
                "category": "theory",
                "keywords": ["capm", "beta", "rendement", "marché", "risque systématique"]
            },
            {
                "title": "Alpha et Beta - Mesures de Performance",
                "content": """Alpha: rendement excédentaire par rapport au benchmark (skill du gérant).
                Beta: sensibilité aux mouvements du marché (exposition au risque systématique).
                Alpha positif = surperformance, Alpha négatif = sous-performance.
                Beta = 1.0 indique que l'actif bouge comme le marché. Beta > 1 amplifie les mouvements.
                Objectif: maximiser alpha (skill) tout en contrôlant beta (exposition risque).""",
                "category": "metrics",
                "keywords": ["alpha", "beta", "performance", "benchmark", "surperformance"]
            },
        ]
        
        for doc in core_docs:
            self.add_document(
                doc['content'],
                {
                    'title': doc['title'],
                    'category': doc['category'],
                    'keywords': doc['keywords'],
                    'source': 'auto-populated',
                    'date_added': datetime.now().isoformat()
                },
                save=False
            )
        
        self.save_index()
        logger.info(f"Auto-populated {len(core_docs)} core documents")
    
    def add_document(self, text: str, metadata: Optional[Dict] = None, save: bool = True):
        """Add document with keyword indexing"""
        if not text.strip():
            return
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Extract keywords for BM25
        keywords = self._extract_keywords(text)
        metadata = metadata or {}
        metadata['auto_keywords'] = keywords
        
        # Store
        doc_id = len(self.documents)
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
        # Update keyword index (for hybrid search)
        self._update_keyword_index(doc_id, keywords)
        
        logger.info(f"Added: {metadata.get('title', 'Untitled')}")
        
        if save:
            self.save_index()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords (simple TF-IDF style)"""
        # Lowercase and tokenize
        words = re.findall(r'\b[a-zàâäéèêëïîôùûüÿæœ]{3,}\b', text.lower())
        
        # Remove common stopwords (French + English)
        stopwords = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'est', 'sont',
            'the', 'a', 'an', 'and', 'or', 'is', 'are', 'in', 'on', 'at', 'to', 'for',
            'que', 'qui', 'dans', 'par', 'pour', 'avec', 'sur', 'plus', 'cette', 'ces'
        }
        
        words = [w for w in words if w not in stopwords]
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_counts.most_common(10)]
    
    def _update_keyword_index(self, doc_id: int, keywords: List[str]):
        """Update inverted index for keyword search"""
        for keyword in keywords:
            if keyword not in self.keywords_index:
                self.keywords_index[keyword] = []
            self.keywords_index[keyword].append(doc_id)
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        query_lower = query.lower()
        expanded = [query]
        
        # Add synonyms from finance terms
        for term, synonyms in self.finance_terms.items():
            if term in query_lower or any(syn in query_lower for syn in synonyms):
                expanded.extend(synonyms)
        
        return list(set(expanded))  # Deduplicate
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25-style keyword search"""
        query_keywords = self._extract_keywords(query)
        
        # Expand query keywords
        expanded_keywords = []
        for kw in query_keywords:
            expanded_keywords.append(kw)
            # Add similar terms
            for term, synonyms in self.finance_terms.items():
                if kw in term or kw in ' '.join(synonyms):
                    expanded_keywords.extend([s.split()[0] for s in synonyms])
        
        expanded_keywords = list(set(expanded_keywords))
        
        # Score documents
        doc_scores = {}
        
        for keyword in expanded_keywords:
            if keyword in self.keywords_index:
                for doc_id in self.keywords_index[keyword]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
        
        # Normalize by document length
        scored_docs = []
        for doc_id, score in doc_scores.items():
            doc_len = len(self.documents[doc_id].split())
            normalized_score = score / (doc_len ** 0.5)  # Length normalization
            scored_docs.append((doc_id, normalized_score))
        
        # Sort and return top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
    
    def search(self, query: str, top_k: int = 3, min_score: float = 0.3, 
               hybrid: bool = True) -> List[Dict]:
        """
        Hybrid search: semantic (embeddings) + keyword (BM25)
        
        Args:
            query: Search query
            top_k: Results to return
            min_score: Minimum similarity (0-1)
            hybrid: Use hybrid search (True) or semantic only (False)
        """
        if not self.embeddings:
            logger.warning("Empty index")
            return []
        
        results = []
        
        if hybrid:
            # Hybrid: combine semantic + keyword
            results = self._hybrid_search(query, top_k, min_score)
        else:
            # Semantic only
            results = self._semantic_search(query, top_k, min_score)
        
        # Re-rank by relevance (financial terms boost)
        results = self._rerank_results(query, results)
        
        return results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int, min_score: float) -> List[Dict]:
        """Pure semantic search with embeddings"""
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding) / (
            np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-top_k * 2:][::-1]  # Get more for reranking
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': score,
                    'type': 'semantic'
                })
        
        return results
    
    def _hybrid_search(self, query: str, top_k: int, min_score: float) -> List[Dict]:
        """Combine semantic + keyword search"""
        # Semantic search
        semantic_results = self._semantic_search(query, top_k * 2, min_score)
        
        # Keyword search
        keyword_matches = self._keyword_search(query, top_k * 2)
        
        # Merge results with weighted scores
        merged = {}
        
        # Add semantic results (weight: 0.7)
        for result in semantic_results:
            doc_idx = self.documents.index(result['text'])
            merged[doc_idx] = {
                'text': result['text'],
                'metadata': result['metadata'],
                'semantic_score': result['score'] * 0.7,
                'keyword_score': 0.0
            }
        
        # Add keyword scores (weight: 0.3)
        for doc_idx, kw_score in keyword_matches:
            if doc_idx in merged:
                merged[doc_idx]['keyword_score'] = kw_score * 0.3
            else:
                merged[doc_idx] = {
                    'text': self.documents[doc_idx],
                    'metadata': self.metadata[doc_idx],
                    'semantic_score': 0.0,
                    'keyword_score': kw_score * 0.3
                }
        
        # Combine scores
        final_results = []
        for doc_idx, data in merged.items():
            combined_score = data['semantic_score'] + data['keyword_score']
            if combined_score >= min_score:
                final_results.append({
                    'text': data['text'],
                    'metadata': data['metadata'],
                    'score': combined_score,
                    'type': 'hybrid'
                })
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-rank results by financial relevance"""
        query_lower = query.lower()
        
        for result in results:
            boost = 0.0
            
            # Boost if query terms appear in title
            title = result['metadata'].get('title', '').lower()
            for word in query_lower.split():
                if len(word) > 3 and word in title:
                    boost += 0.1
            
            # Boost if category matches query intent
            category = result['metadata'].get('category', '')
            if 'risk' in query_lower and category == 'risk':
                boost += 0.15
            elif any(w in query_lower for w in ['ratio', 'sharpe', 'sortino']) and category == 'metrics':
                boost += 0.15
            elif any(w in query_lower for w in ['diversif', 'allocation']) and category == 'strategy':
                boost += 0.15
            
            # Apply boost
            result['score'] = min(1.0, result['score'] + boost)
        
        # Re-sort
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def add_from_web_search(self, topic: str, web_results: Dict):
        """
        Auto-expand knowledge base from web search results
        
        Args:
            topic: Topic searched
            web_results: Results from WebSearchEngine
        """
        # Extract from Wikipedia
        wiki = web_results.get('sources', {}).get('wikipedia', {})
        if wiki.get('found'):
            self.add_document(
                wiki['summary'],
                {
                    'title': f"{wiki['title']} (Wikipedia)",
                    'category': 'web-sourced',
                    'source': 'wikipedia',
                    'url': wiki['url'],
                    'language': wiki['language'],
                    'date_added': datetime.now().isoformat()
                },
                save=False
            )
            logger.info(f"Added from Wikipedia: {wiki['title']}")
        
        # Extract from DuckDuckGo top results
        ddg = web_results.get('sources', {}).get('duckduckgo', {})
        if ddg.get('results'):
            for result in ddg['results'][:2]:  # Top 2 results
                if result.get('snippet'):
                    self.add_document(
                        result['snippet'],
                        {
                            'title': result['title'],
                            'category': 'web-sourced',
                            'source': 'duckduckgo',
                            'url': result['url'],
                            'date_added': datetime.now().isoformat()
                        },
                        save=False
                    )
        
        self.save_index()
        logger.info(f"Knowledge base expanded with web results for: {topic}")
    
    # Original methods (unchanged)
    def add_documents_from_folder(self, folder_path: Optional[str] = None):
        """Load documents from folder"""
        folder = Path(folder_path) if folder_path else self.documents_dir
        if not folder.exists():
            return
        
        files = [f for f in folder.iterdir() if f.suffix in ['.txt', '.md', '.json']]
        for file_path in files:
            try:
                self._load_file(file_path)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        self.save_index()
    
    def _load_file(self, file_path: Path):
        """Load single file"""
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get('content', '')
                metadata = {
                    'title': data.get('title', file_path.stem),
                    'source': str(file_path),
                    'category': data.get('category', 'general')
                }
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                metadata = {
                    'title': file_path.stem,
                    'source': str(file_path),
                    'category': 'general'
                }
        
        chunks = self._split_text(text, max_length=512)
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            if len(chunks) > 1:
                chunk_metadata['chunk'] = i + 1
            self.add_document(chunk, chunk_metadata, save=False)
    
    def _split_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks"""
        sentences = text.replace('\n', ' ').split('. ')
        chunks, current = [], []
        current_len = 0
        
        for sentence in sentences:
            sent_len = len(sentence.split())
            if current_len + sent_len > max_length and current:
                chunks.append('. '.join(current) + '.')
                current, current_len = [sentence], sent_len
            else:
                current.append(sentence)
                current_len += sent_len
        
        if current:
            chunks.append('. '.join(current) + '.')
        
        return chunks or [text]
    
    def save_index(self):
        """Save index with keyword index"""
        index_path = self.embeddings_dir / "rag_index.pkl"
        index_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'keywords_index': self.keywords_index,
            'version': '2.0'  # Version with hybrid search
        }
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        logger.info(f"Saved index: {len(self.documents)} docs")
    
    def load_index(self):
        """Load index with keyword index"""
        index_path = self.embeddings_dir / "rag_index.pkl"
        if not index_path.exists():
            return
        
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data.get('documents', [])
            self.embeddings = index_data.get('embeddings', [])
            self.metadata = index_data.get('metadata', [])
            self.keywords_index = index_data.get('keywords_index', {})
            
            logger.info(f"Loaded: {len(self.documents)} docs (v{index_data.get('version', '1.0')})")
        except Exception as e:
            logger.error(f"Load error: {e}")
    
    def clear_index(self):
        """Clear index"""
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.keywords_index = {}
        self.save_index()
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        categories = {}
        for meta in self.metadata:
            cat = meta.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'categories': categories,
            'indexed_keywords': len(self.keywords_index),
            'model': self.model._modules['0'].auto_model.name_or_path,
            'version': '2.0-hybrid'
        }


# Quick search helper
def quick_search(query: str) -> List[Dict]:
    rag = SimpleRAG()
    return rag.search(query, top_k=3, hybrid=True)


if __name__ == "__main__":
    print("🔧 Testing Enhanced RAG\n")
    
    rag = SimpleRAG()
    stats = rag.get_stats()
    
    print(f"📊 Stats:")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Keywords indexed: {stats['indexed_keywords']}")
    print(f"  Categories: {stats['categories']}")
    
    print("\n🔍 Testing hybrid search...")
    queries = [
        "Qu'est-ce que le ratio de Sharpe?",
        "Comment mesurer le risque d'un portfolio?",
        "Expliquer la diversification"
    ]
    
    for query in queries:
        print(f"\n  Query: {query}")
        results = rag.search(query, top_k=2, hybrid=True)
        for i, r in enumerate(results, 1):
            print(f"    {i}. {r['metadata']['title']} (score: {r['score']:.3f}, type: {r['type']})")
    
    print("\n✅ Enhanced RAG test complete!")
