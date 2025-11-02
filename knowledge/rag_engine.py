"""
Enhanced RAG Engine v3.0 - Finance Optimized
==============================================
New Features:
- PDF support with intelligent chunking
- Smart overlap for context preservation
- Cross-encoder re-ranking (optional)
- Advanced query expansion
- Intent detection
- Batch processing
- Better statistics
"""

import json
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
import pickle
import logging
import re
from collections import Counter
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from PyPDF2 import PdfReader
    from pdfminer.high_level import extract_text as pdf_extract_text
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("⚠️ PDF support unavailable. Install: pip install PyPDF2 pdfminer.six")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("⚠️ Cross-encoder unavailable (optional). Install: pip install sentence-transformers")


class SimpleRAG:
    """Enhanced RAG with PDF support, hybrid search, and advanced features"""
    
    def __init__(self, 
                 knowledge_dir: str = "knowledge", 
                 model_name: str = "BAAI/bge-small-en-v1.5",
                 use_cross_encoder: bool = True):
        """
        Initialize RAG Engine
        
        Args:
            knowledge_dir: Directory for storage
            model_name: Sentence transformer model
                - "all-MiniLM-L6-v2" (90MB, fast, good)
                - "BAAI/bge-small-en-v1.5" (130MB, fast, excellent for finance)
                - "all-mpnet-base-v2" (420MB, slower, very good)
            use_cross_encoder: Enable re-ranking (better quality, +0.3s)
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.documents_dir = self.knowledge_dir / "documents"
        self.embeddings_dir = self.knowledge_dir / "embeddings"
        self.pdf_dir = self.knowledge_dir / "pdfs"
        
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE
        
        # Lazy loading
        self._model = None
        self._cross_encoder = None
        
        # Storage
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.keywords_index = {}
        self.term_frequencies = {}
        
        # Financial terms dictionary
        self.finance_terms = self._load_finance_terms()
        
        self.load_index()
        
        # Auto-populate if empty
        if len(self.documents) == 0:
            logger.info("📚 Index empty, auto-populating with core concepts...")
            self._auto_populate()
        
        logger.info(f"✅ RAG Engine initialized: {len(self.documents)} docs")
    
    @property
    def model(self):
        """Lazy load embedding model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded")
        return self._model
    
    @property
    def cross_encoder(self):
        """Lazy load cross-encoder for re-ranking"""
        if self._cross_encoder is None and self.use_cross_encoder:
            logger.info("Loading cross-encoder for re-ranking...")
            try:
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("✅ Cross-encoder loaded")
            except Exception as e:
                logger.warning(f"Cross-encoder failed: {e}")
                self.use_cross_encoder = False
        return self._cross_encoder
    
    def _load_finance_terms(self) -> Dict[str, List[str]]:
        """Financial terms and synonyms for query expansion"""
        return {
            'sharpe': ['sharpe ratio', 'rendement ajusté au risque', 'risk-adjusted return'],
            'var': ['value at risk', 'var', 'valeur à risque', 'perte maximale'],
            'cvar': ['conditional var', 'expected shortfall', 'cvar'],
            'sortino': ['sortino ratio', 'downside risk', 'risque à la baisse'],
            'markowitz': ['markowitz', 'modern portfolio theory', 'mpt', 'frontière efficiente', 'efficient frontier'],
            'diversification': ['diversification', 'répartition', 'allocation', 'spread risk'],
            'volatility': ['volatilité', 'volatility', 'écart-type', 'standard deviation', 'risque'],
            'capm': ['capm', 'capital asset pricing model', 'modèle d\'évaluation des actifs'],
            'black-litterman': ['black-litterman', 'bl model', 'equilibrium', 'bayesian'],
            'beta': ['beta', 'bêta', 'sensibilité au marché', 'market sensitivity'],
            'alpha': ['alpha', 'surperformance', 'outperformance', 'excess return'],
            'portfolio': ['portfolio', 'portefeuille', 'asset allocation'],
            'risk': ['risk', 'risque', 'uncertainty', 'incertitude'],
            'return': ['return', 'rendement', 'performance', 'profit'],
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
                Limite: suppose distribution normale des rendements. Utilisé dans PyManager pour évaluer les portfolios optimisés.
                Plus le Sharpe est élevé, meilleur est le rendement par unité de risque pris.""",
                "category": "metrics",
                "keywords": ["sharpe", "ratio", "risque", "rendement", "performance", "volatilité"]
            },
            {
                "title": "Value at Risk (VaR) - Mesure du Risque",
                "content": """La VaR estime la perte maximale probable d'un portfolio sur une période avec un niveau de confiance donné.
                Exemple: VaR 95% = 10K€ signifie 95% de chances que perte ≤ 10K€.
                Trois méthodes: 1) Historique (données passées), 2) Paramétrique (distribution normale), 3) Monte Carlo (simulations).
                Limite: ne dit rien au-delà du seuil. Compléter avec CVaR (Expected Shortfall).
                Dans PyManager: disponible dans Portfolio Details > Analytics.
                Utilisée par les régulateurs bancaires (Bâle III) pour mesurer l'exposition au risque.""",
                "category": "risk",
                "keywords": ["var", "value at risk", "perte", "risque", "probabilité", "cvar"]
            },
            {
                "title": "Ratio de Sortino - Risque Asymétrique",
                "content": """Le ratio de Sortino améliore le Sharpe en ne pénalisant que la volatilité négative (downside).
                Formule: (Rendement - Taux cible) / Écart-type downside.
                Avantage: ne pénalise pas les hausses, seulement les baisses. Plus adapté pour investisseurs averses aux pertes.
                Le Sharpe pénalise toute volatilité, le Sortino uniquement le downside. Meilleur pour stratégies asymétriques.
                Particulièrement utile pour les hedge funds et stratégies alternatives.""",
                "category": "metrics",
                "keywords": ["sortino", "downside", "risque", "asymétrique", "baisse"]
            },
            {
                "title": "Théorie Moderne du Portfolio (Markowitz)",
                "content": """Développée par Harry Markowitz (1952), optimise le ratio rendement/risque par diversification.
                Concept: Frontière efficiente = ensemble des portfolios optimaux (max rendement pour risque donné).
                Optimisation: maximiser Sharpe sous contraintes (somme poids = 100%, poids ≥ 0%).
                PyManager propose 4 modes: Sharp (max Sharpe), Risk (min volatilité), Return (max rendement), Unsafe (sans contraintes).
                Limites: suppose rendements normaux, basé sur historique, sensible aux estimations.
                Prix Nobel d'économie 1990. Révolution dans la gestion d'actifs moderne.""",
                "category": "theory",
                "keywords": ["markowitz", "mpt", "frontière", "efficiente", "optimisation", "diversification"]
            },
            {
                "title": "Modèle Black-Litterman",
                "content": """Combine équilibre de marché + vues personnelles pour allocation d'actifs.
                Processus: 1) Partir équilibre marché (cap-weighted), 2) Ajouter vos vues avec confiance, 3) Ajuster poids.
                Avantages vs Markowitz: moins sensible aux estimations, poids plus stables, intègre insights de l'investisseur.
                Développé par Fischer Black et Robert Litterman (1990) chez Goldman Sachs.
                Dans PyManager: Build Portfolio > Black-Litterman avec interface de saisie de vues.
                Utilisé par les gestionnaires d'actifs institutionnels pour allocation tactique.""",
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
                Dans PyManager: vérifier Sector Allocation dans Portfolio Details.
                La diversification est le seul "free lunch" en finance (Harry Markowitz).""",
                "category": "strategy",
                "keywords": ["diversification", "corrélation", "allocation", "secteurs", "risque"]
            },
            {
                "title": "CAPM - Capital Asset Pricing Model",
                "content": """Modèle d'évaluation des actifs financiers basé sur le risque systématique (bêta).
                Formule: Rendement attendu = Taux sans risque + β × (Rendement marché - Taux sans risque).
                Beta: β=1 (comme marché), β>1 (plus volatil), β<1 (moins volatil).
                Exemple: AAPL β=1.2, Rf=2%, Rm=10% → Rendement attendu = 2% + 1.2×8% = 11.6%.
                Limites: suppose marchés efficients, un seul facteur. Alternative: Fama-French (3 facteurs).
                Développé par William Sharpe (1964), Prix Nobel 1990.""",
                "category": "theory",
                "keywords": ["capm", "beta", "rendement", "marché", "risque systématique"]
            },
            {
                "title": "Alpha et Beta - Mesures de Performance",
                "content": """Alpha: rendement excédentaire par rapport au benchmark (skill du gérant).
                Beta: sensibilité aux mouvements du marché (exposition au risque systématique).
                Alpha positif = surperformance, Alpha négatif = sous-performance.
                Beta = 1.0 indique que l'actif bouge comme le marché. Beta > 1 amplifie les mouvements.
                Objectif: maximiser alpha (skill) tout en contrôlant beta (exposition risque).
                Jensen's alpha = mesure de surperformance ajustée au risque.""",
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
        logger.info(f"✅ Auto-populated {len(core_docs)} core documents")
    
    # =============================================================================
    # PDF SUPPORT (NEW)
    # =============================================================================
    
    def add_pdf(self, pdf_path: str, category: str = 'pdf', extract_tables: bool = False):
        """
        Extract and index PDF content with intelligent chunking
        
        Args:
            pdf_path: Path to PDF file
            category: Category for organization
            extract_tables: Extract tables (requires tabula-py, experimental)
        """
        if not PDF_AVAILABLE:
            logger.error("❌ PDF support not installed. Run: pip install PyPDF2 pdfminer.six")
            return
        
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"❌ PDF not found: {pdf_path}")
                return
            
            logger.info(f"📄 Processing PDF: {pdf_path.name}")
            
            # Extract text (pdfminer is better than PyPDF2)
            try:
                text = pdf_extract_text(str(pdf_path))
            except:
                # Fallback to PyPDF2
                reader = PdfReader(str(pdf_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            if not text.strip():
                logger.warning(f"⚠️ No text extracted from {pdf_path.name}")
                return
            
            # Chunk intelligently with overlap
            chunks = self._split_pdf_text(text, max_length=512, overlap=50)
            
            logger.info(f"  Split into {len(chunks)} chunks")
            
            # Add each chunk
            added_count = 0
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                self.add_document(
                    chunk,
                    {
                        'title': f"{pdf_path.stem} - Part {i+1}/{len(chunks)}",
                        'source': str(pdf_path),
                        'category': category,
                        'chunk_index': i + 1,
                        'total_chunks': len(chunks),
                        'type': 'pdf',
                        'filename': pdf_path.name,
                        'date_added': datetime.now().isoformat()
                    },
                    save=False
                )
                added_count += 1
            
            self.save_index()
            logger.info(f"✅ PDF indexed: {pdf_path.name} ({added_count} chunks)")
            
        except Exception as e:
            logger.error(f"❌ PDF processing error: {e}")
    
    def _split_pdf_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """
        Smart PDF splitting with overlap for context preservation
        
        Args:
            text: Full text to split
            max_length: Max words per chunk
            overlap: Words to overlap between chunks (preserves context)
        """
        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r' {2,}', ' ', text)  # Remove excessive spaces
        text = re.sub(r'-\n', '', text)  # Remove hyphenation at line breaks
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            words = para.split()
            para_len = len(words)
            
            if current_len + para_len > max_length and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last N words from previous)
                all_words = ' '.join(current_chunk).split()
                overlap_words = all_words[-overlap:] if len(all_words) > overlap else all_words
                
                current_chunk = overlap_words + words
                current_len = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_len += para_len
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def add_pdfs_from_folder(self, folder_path: str, recursive: bool = True, category: str = 'pdf'):
        """
        Batch process PDFs from folder
        
        Args:
            folder_path: Folder containing PDFs
            recursive: Search in subfolders
            category: Category for all PDFs
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.error(f"❌ Folder not found: {folder_path}")
            return
        
        # Find PDFs
        if recursive:
            pdf_files = list(folder.rglob("*.pdf"))
        else:
            pdf_files = list(folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️ No PDFs found in {folder_path}")
            return
        
        logger.info(f"📚 Found {len(pdf_files)} PDFs to process")
        
        success_count = 0
        for pdf_path in pdf_files:
            try:
                self.add_pdf(str(pdf_path), category=category)
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Failed {pdf_path.name}: {e}")
        
        logger.info(f"✅ Processed {success_count}/{len(pdf_files)} PDFs successfully")
    
    # =============================================================================
    # DOCUMENT MANAGEMENT
    # =============================================================================
    
    def add_document(self, text: str, metadata: Optional[Dict] = None, save: bool = True):
        """
        Add document with deduplication and keyword indexing
        
        Args:
            text: Document content
            metadata: Document metadata
            save: Save index immediately
        """
        if not text.strip():
            return
        
        # Check if already indexed (deduplication)
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if any(m.get('text_hash') == text_hash for m in self.metadata):
            logger.debug(f"⊘ Already indexed (skipping): {metadata.get('title', 'Untitled')}")
            return
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        
        # Extract keywords for hybrid search
        keywords = self._extract_keywords(text)
        
        metadata = metadata or {}
        metadata['auto_keywords'] = keywords
        metadata['text_hash'] = text_hash
        
        # Store
        doc_id = len(self.documents)
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
        # Update keyword index
        self._update_keyword_index(doc_id, keywords)
        
        logger.debug(f"✓ Added: {metadata.get('title', 'Untitled')}")
        
        if save:
            self.save_index()
    
    def add_documents_batch(self, texts: List[str], metadatas: List[Dict]):
        """
        Add multiple documents at once (faster than individual adds)
        
        Args:
            texts: List of document contents
            metadatas: List of metadata dicts (same length as texts)
        """
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have same length")
        
        logger.info(f"📚 Batch encoding {len(texts)} documents...")
        
        # Batch encode (much faster)
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            # Check deduplication
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if any(m.get('text_hash') == text_hash for m in self.metadata):
                continue
            
            keywords = self._extract_keywords(text)
            metadata['auto_keywords'] = keywords
            metadata['text_hash'] = text_hash
            
            doc_id = len(self.documents)
            self.documents.append(text)
            self.embeddings.append(embedding)
            self.metadata.append(metadata)
            
            self._update_keyword_index(doc_id, keywords)
        
        self.save_index()
        logger.info(f"✅ Batch added {len(texts)} documents")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords (TF-IDF style)"""
        # Lowercase and tokenize
        words = re.findall(r'\b[a-zàâäéèêëïîôùûüÿæœ]{3,}\b', text.lower())
        
        # Remove stopwords
        stopwords = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'est', 'sont',
            'the', 'a', 'an', 'and', 'or', 'is', 'are', 'in', 'on', 'at', 'to', 'for',
            'que', 'qui', 'dans', 'par', 'pour', 'avec', 'sur', 'plus', 'cette', 'ces',
            'être', 'avoir', 'faire', 'dire', 'peut', 'dont', 'très', 'aussi', 'bien'
        }
        
        words = [w for w in words if w not in stopwords and len(w) > 3]
        
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
    
    # =============================================================================
    # SEARCH ENGINE
    # =============================================================================
    
    def search(self, 
               query: str, 
               top_k: int = 3, 
               min_score: float = 0.3, 
               hybrid: bool = True,
               category_filter: Optional[str] = None,
               source_filter: Optional[str] = None) -> List[Dict]:
        """
        Hybrid search with advanced features
        
        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity (0-1)
            hybrid: Use hybrid (semantic + keyword) search
            category_filter: Filter by category ('risk', 'metrics', 'theory', etc.)
            source_filter: Filter by source ('pdf', 'auto-populated', etc.)
        
        Returns:
            List of results with text, metadata, and scores
        """
        if not self.embeddings:
            logger.warning("⚠️ Empty index, cannot search")
            return []
        
        # Get initial results
        if hybrid:
            results = self._hybrid_search(query, top_k * 3, min_score)
        else:
            results = self._semantic_search(query, top_k * 3, min_score)
        
        # Apply filters
        if category_filter:
            results = [r for r in results if r['metadata'].get('category') == category_filter]
        
        if source_filter:
            results = [r for r in results if source_filter in str(r['metadata'].get('source', ''))]
        
        # Re-rank
        if self.use_cross_encoder and len(results) > 1:
            results = self._rerank_with_cross_encoder(query, results)
        else:
            results = self._rerank_results(query, results)
        
        return results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int, min_score: float) -> List[Dict]:
        """Pure semantic search with embeddings"""
        # Expand query
        expanded_queries = self._expand_query(query)
        
        # Encode queries
        query_embeddings = [self.model.encode(q, normalize_embeddings=True) for q in expanded_queries]
        
        # Average embeddings (query expansion)
        query_embedding = np.mean(query_embeddings, axis=0)
        
        # Calculate similarities
        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding)
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k * 2:][::-1]
        
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
        
        # Merge results
        merged = {}
        
        # Add semantic (weight: 0.7)
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
        
        # Sort by score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25-style keyword search"""
        query_keywords = self._extract_keywords(query)
        
        # Expand keywords
        expanded_keywords = []
        for kw in query_keywords:
            expanded_keywords.append(kw)
            # Add similar terms from finance dictionary
            for term, synonyms in self.finance_terms.items():
                if kw in term or kw in ' '.join(synonyms):
                    expanded_keywords.extend([s.split()[0] for s in synonyms])
        
        expanded_keywords = list(set(expanded_keywords))
        
        # Score documents
        doc_scores = {}
        
        for keyword in expanded_keywords:
            if keyword in self.keywords_index:
                for doc_idx in self.keywords_index[keyword]:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1
        
        # Normalize by document length
        scored_docs = []
        for doc_idx, score in doc_scores.items():
            doc_len = len(self.documents[doc_idx].split())
            normalized_score = score / (doc_len ** 0.5)
            scored_docs.append((doc_idx, normalized_score))
        
        # Sort and return top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Advanced query expansion with contextual synonyms
        """
        query_lower = query.lower()
        expanded = [query]
        
        # 1. Finance terms expansion
        for term, synonyms in self.finance_terms.items():
            if term in query_lower or any(syn in query_lower for syn in synonyms):
                expanded.extend(synonyms)
        
        # 2. Contextual expansion based on query type
        if any(w in query_lower for w in ['comment', 'how', 'explain', 'explique']):
            expanded.append(f"{query} tutorial")
            expanded.append(f"{query} example")
        
        elif any(w in query_lower for w in ['calcul', 'calculate', 'formule', 'formula']):
            expanded.append(f"{query} calculation")
            expanded.append(f"{query} equation")
        
        elif any(w in query_lower for w in ['différence', 'difference', 'vs', 'compare']):
            expanded.append(f"{query} comparison")
            expanded.append(f"{query} versus")
        
        # 3. Acronym expansion
        acronym_map = {
            'var': 'value at risk',
            'cvar': 'conditional value at risk',
            'capm': 'capital asset pricing model',
            'mpt': 'modern portfolio theory',
            'roi': 'return on investment',
            'irr': 'internal rate of return',
            'wacc': 'weighted average cost of capital',
        }
        
        for acronym, full_name in acronym_map.items():
            if re.search(rf'\b{acronym}\b', query_lower):
                expanded.append(full_name)
        
        return list(set(expanded))
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Basic re-ranking by financial relevance"""
        query_lower = query.lower()
        
        for result in results:
            boost = 0.0
            
            # Boost if query terms in title
            title = result['metadata'].get('title', '').lower()
            for word in query_lower.split():
                if len(word) > 3 and word in title:
                    boost += 0.1
            
            # Boost by category match
            category = result['metadata'].get('category', '')
            if 'risk' in query_lower and category == 'risk':
                boost += 0.15
            elif any(w in query_lower for w in ['ratio', 'sharpe', 'sortino']) and category == 'metrics':
                boost += 0.15
            elif any(w in query_lower for w in ['diversif', 'allocation']) and category == 'strategy':
                boost += 0.15
            
            # Boost for exact keyword matches
            keywords = result['metadata'].get('auto_keywords', [])
            for word in query_lower.split():
                if word in keywords:
                    boost += 0.05
            
            result['score'] = min(1.0, result['score'] + boost)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _rerank_with_cross_encoder(self, query: str, results: List[Dict]) -> List[Dict]:
        """Advanced re-ranking with cross-encoder"""
        if not results or not self.use_cross_encoder:
            return results
        
        try:
            # Prepare pairs
            pairs = [(query, r['text']) for r in results]
            
            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update scores (weighted combination)
            for i, score in enumerate(scores):
                results[i]['cross_encoder_score'] = float(score)
                # 60% original + 40% cross-encoder
                results[i]['score'] = 0.6 * results[i]['score'] + 0.4 * float(score)
            
            # Re-sort
            results.sort(key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.warning(f"Cross-encoder re-ranking failed: {e}")
        
        return results
    
    # =============================================================================
    # INTENT DETECTION (NEW)
    # =============================================================================
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect query intent for smart routing
        
        Returns:
            {
                'intent': 'definition' | 'calculation' | 'comparison' | 'example' | 'general',
                'confidence': float,
                'entities': List[str],
                'suggested_filters': Dict
            }
        """
        query_lower = query.lower()
        
        intent = 'general'
        confidence = 0.5
        entities = []
        suggested_filters = {}
        
        # Definition intent
        if any(w in query_lower for w in ['qu\'est-ce', 'what is', 'définition', 'define', 'explique', 'explain']):
            intent = 'definition'
            confidence = 0.9
            suggested_filters['category'] = 'theory'
        
        # Calculation intent
        elif any(w in query_lower for w in ['calcul', 'calculate', 'formule', 'formula', 'comment calculer']):
            intent = 'calculation'
            confidence = 0.85
            suggested_filters['category'] = 'metrics'
        
        # Comparison intent
        elif any(w in query_lower for w in ['différence', 'difference', 'vs', 'versus', 'compare', 'comparer']):
            intent = 'comparison'
            confidence = 0.9
        
        # Example intent
        elif any(w in query_lower for w in ['exemple', 'example', 'case study', 'illustration', 'cas pratique']):
            intent = 'example'
            confidence = 0.8
        
        # Extract financial entities
        for term in self.finance_terms.keys():
            if term in query_lower:
                entities.append(term)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'suggested_filters': suggested_filters
        }
    
    # =============================================================================
    # FILE MANAGEMENT
    # =============================================================================
    
    def add_documents_from_folder(self, folder_path: Optional[str] = None):
        """Load documents from folder (txt, md, json)"""
        folder = Path(folder_path) if folder_path else self.documents_dir
        if not folder.exists():
            logger.warning(f"⚠️ Folder not found: {folder}")
            return
        
        files = [f for f in folder.iterdir() if f.suffix in ['.txt', '.md', '.json']]
        
        if not files:
            logger.info(f"No documents found in {folder}")
            return
        
        logger.info(f"📚 Loading {len(files)} files from {folder}")
        
        for file_path in files:
            try:
                self._load_file(file_path)
            except Exception as e:
                logger.error(f"❌ Error loading {file_path.name}: {e}")
        
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
        
        chunks = self._split_text(text, max_length=512, overlap=50)
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            if len(chunks) > 1:
                chunk_metadata['chunk'] = i + 1
                chunk_metadata['total_chunks'] = len(chunks)
            self.add_document(chunk, chunk_metadata, save=False)
    
    def _split_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """Split text with semantic overlap"""
        # Tokenize into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            words = sentence.split()
            sent_len = len(words)
            
            if current_len + sent_len > max_length and current_chunk:
                # Save chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new with overlap
                overlap_words = ' '.join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words + words
                current_len = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_len += sent_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    # =============================================================================
    # INDEX MANAGEMENT
    # =============================================================================
    
    def save_index(self):
        """Save index with all data"""
        index_path = self.embeddings_dir / "rag_index.pkl"
        index_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'keywords_index': self.keywords_index,
            'version': '3.0',
            'model_name': self.model_name,
            'use_cross_encoder': self.use_cross_encoder
        }
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        logger.debug(f"💾 Saved index: {len(self.documents)} docs")
    
    def load_index(self):
        """Load index"""
        index_path = self.embeddings_dir / "rag_index.pkl"
        if not index_path.exists():
            logger.info("No existing index found")
            return
        
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data.get('documents', [])
            self.embeddings = index_data.get('embeddings', [])
            self.metadata = index_data.get('metadata', [])
            self.keywords_index = index_data.get('keywords_index', {})
            
            logger.info(f"📚 Loaded index: {len(self.documents)} docs (v{index_data.get('version', '1.0')})")
        except Exception as e:
            logger.error(f"❌ Load error: {e}")
    
    def clear_index(self):
        """Clear all data"""
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.keywords_index = {}
        self.save_index()
        logger.info("🗑️ Index cleared")
    
    # =============================================================================
    # STATISTICS
    # =============================================================================
    
    def get_stats(self) -> Dict:
        """Enhanced statistics"""
        categories = {}
        sources = {}
        keywords_dist = Counter()
        
        for meta in self.metadata:
            # Categories
            cat = meta.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Sources
            source_type = meta.get('source', 'unknown')
            if 'pdf' in str(source_type).lower():
                source_type = 'pdf'
            elif 'auto' in str(source_type).lower():
                source_type = 'auto-populated'
            elif 'web' in str(source_type).lower():
                source_type = 'web-sourced'
            else:
                source_type = 'manual'
            
            sources[source_type] = sources.get(source_type, 0) + 1
            
            # Keywords
            for kw in meta.get('auto_keywords', []):
                keywords_dist[kw] += 1
        
        return {
            'total_documents': len(self.documents),
            'categories': categories,
            'sources': sources,
            'indexed_keywords': len(self.keywords_index),
            'top_keywords': dict(keywords_dist.most_common(20)),
            'model': self.model_name,
            'version': '3.0-enhanced',
            'avg_doc_length': int(np.mean([len(d.split()) for d in self.documents])) if self.documents else 0,
            'reranking': 'cross-encoder' if self.use_cross_encoder else 'keyword-based',
            'pdf_support': PDF_AVAILABLE
        }
    
    # =============================================================================
    # WEB INTEGRATION
    # =============================================================================
    
    def add_from_web_search(self, topic: str, web_results: Dict):
        """Auto-expand knowledge base from web search results"""
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
            logger.info(f"✓ Added from Wikipedia: {wiki['title']}")
        
        # Extract from DuckDuckGo top results
        ddg = web_results.get('sources', {}).get('duckduckgo', {})
        if ddg.get('results'):
            for result in ddg['results'][:2]:
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
        logger.info(f"✅ Knowledge base expanded with web results for: {topic}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def quick_search(query: str) -> List[Dict]:
    """Quick search helper"""
    rag = SimpleRAG()
    return rag.search(query, top_k=3, hybrid=True)


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    print("🔧 Testing Enhanced RAG v3.0\n")
    print("="*70)
    
    rag = SimpleRAG()
    stats = rag.get_stats()
    
    print(f"\n📊 Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Keywords indexed: {stats['indexed_keywords']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Sources: {stats['sources']}")
    print(f"  Model: {stats['model']}")
    print(f"  Re-ranking: {stats['reranking']}")
    print(f"  PDF support: {'✅' if stats['pdf_support'] else '❌'}")
    
    # Test search
    print(f"\n{'='*70}")
    print("🔍 Testing hybrid search...")
    print('='*70)
    
    queries = [
        "Qu'est-ce que le ratio de Sharpe?",
        "Comment calculer la VaR?",
        "Différence entre Sharpe et Sortino"
    ]
    
    for query in queries:
        print(f"\n  Query: {query}")
        
        # Intent detection
        intent = rag.analyze_query_intent(query)
        print(f"  Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
        
        # Search
        results = rag.search(query, top_k=2, hybrid=True)
        
        for i, r in enumerate(results, 1):
            print(f"    {i}. {r['metadata']['title']}")
            print(f"       Score: {r['score']:.3f} | Type: {r['type']}")
            print(f"       Preview: {r['text'][:100]}...")
    
    print(f"\n{'='*70}")
    print("✅ Enhanced RAG v3.0 test complete!")
    print('='*70)
