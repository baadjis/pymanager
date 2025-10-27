# knowledge/rag_engine.py
"""
RAG Engine - Retrieval-Augmented Generation
Simple implementation using sentence-transformers (CPU-friendly)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRAG:
    """
    Simple RAG engine for financial documents
    Uses sentence-transformers for embeddings (no GPU needed)
    """
    
    def __init__(self, knowledge_dir: str = "knowledge", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG engine
        
        Args:
            knowledge_dir: Directory containing documents
            model_name: Sentence transformer model (lightweight by default)
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.documents_dir = self.knowledge_dir / "documents"
        self.embeddings_dir = self.knowledge_dir / "embeddings"
        
        # Create directories if they don't exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model (lightweight, CPU-friendly)
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Storage
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # Load existing index
        self.load_index()
    
    def add_document(self, text: str, metadata: Optional[Dict] = None, save: bool = True):
        """
        Add a document to the RAG index
        
        Args:
            text: Document text
            metadata: Optional metadata (title, source, category, etc.)
            save: Save index after adding
        """
        if not text.strip():
            return
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Store
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
        
        logger.info(f"Added document: {metadata.get('title', 'Untitled')}")
        
        if save:
            self.save_index()
    
    def add_documents_from_folder(self, folder_path: Optional[str] = None):
        """
        Load all documents from a folder
        Supports: .txt, .md, .json
        """
        folder = Path(folder_path) if folder_path else self.documents_dir
        
        if not folder.exists():
            logger.warning(f"Folder not found: {folder}")
            return
        
        supported_extensions = ['.txt', '.md', '.json']
        files = [f for f in folder.iterdir() if f.suffix in supported_extensions]
        
        logger.info(f"Found {len(files)} documents in {folder}")
        
        for file_path in files:
            try:
                self._load_file(file_path)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        self.save_index()
    
    def _load_file(self, file_path: Path):
        """Load a single file"""
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
        
        # Split into chunks if too long (max 512 tokens)
        chunks = self._split_text(text, max_length=512)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            if len(chunks) > 1:
                chunk_metadata['chunk'] = i + 1
                chunk_metadata['total_chunks'] = len(chunks)
            
            self.add_document(chunk, chunk_metadata, save=False)
    
    def _split_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks"""
        # Simple splitting by sentences
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks if chunks else [text]
    
    def search(self, query: str, top_k: int = 3, min_score: float = 0.3) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of documents with metadata and scores
        """
        if not self.embeddings:
            logger.warning("No documents in index")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Calculate cosine similarity
        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding) / (
            np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by minimum score
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': score
                })
        
        return results
    
    def save_index(self):
        """Save index to disk"""
        index_path = self.embeddings_dir / "rag_index.pkl"
        
        index_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Saved index with {len(self.documents)} documents")
    
    def load_index(self):
        """Load index from disk"""
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
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
    
    def clear_index(self):
        """Clear all documents from index"""
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.save_index()
        logger.info("Index cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        categories = {}
        for meta in self.metadata:
            cat = meta.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'categories': categories,
            'model': self.model._modules['0'].auto_model.name_or_path
        }


# =============================================================================
# Helper Functions
# =============================================================================

def create_sample_documents():
    """Create sample financial documents for testing"""
    
    documents = [
        {
            "title": "Ratio de Sharpe",
            "content": """Le ratio de Sharpe est une mesure du rendement ajusté au risque. 
            Il est calculé en divisant le rendement excédentaire (rendement du portfolio moins le taux sans risque) 
            par l'écart-type du portfolio. Un ratio de Sharpe supérieur à 1 est considéré comme bon, 
            supérieur à 2 comme très bon, et supérieur à 3 comme excellent. 
            Le ratio aide les investisseurs à comprendre le rendement d'un investissement par rapport à son risque.""",
            "category": "metrics"
        },
        {
            "title": "Diversification",
            "content": """La diversification est une stratégie de gestion des risques qui mélange 
            une grande variété d'investissements dans un portfolio. L'idée est qu'un portfolio 
            construit avec différents types d'investissements aura, en moyenne, des rendements 
            plus élevés et un risque plus faible qu'un investissement individuel. 
            La diversification s'applique à travers les classes d'actifs, les secteurs et les géographies.""",
            "category": "strategy"
        },
        {
            "title": "Modèle de Markowitz",
            "content": """La théorie moderne du portfolio, développée par Harry Markowitz en 1952, 
            est un cadre mathématique pour assembler un portfolio d'actifs de manière à maximiser 
            le rendement attendu pour un niveau de risque donné. La théorie repose sur l'idée que 
            le risque et le rendement sont directement liés et que les investisseurs sont rationnels 
            et averses au risque. Le modèle utilise la variance comme mesure du risque.""",
            "category": "theory"
        },
        {
            "title": "Black-Litterman",
            "content": """Le modèle Black-Litterman est une approche d'allocation d'actifs 
            qui permet aux gestionnaires de portfolio d'incorporer leurs vues sur les rendements futurs 
            tout en utilisant l'équilibre du marché comme point de départ. Développé par Fischer Black 
            et Robert Litterman en 1990, ce modèle résout certains problèmes pratiques du modèle 
            de Markowitz en produisant des allocations plus stables et intuitives.""",
            "category": "theory"
        },
        {
            "title": "Value at Risk (VaR)",
            "content": """La Value at Risk (VaR) est une mesure statistique du risque de perte 
            d'un investissement. Elle estime la perte maximale qu'un portfolio pourrait subir 
            sur une période donnée avec un niveau de confiance spécifique. Par exemple, 
            une VaR journalière de 1 million de dollars à 95% signifie qu'il y a 5% de chance 
            que le portfolio perde plus de 1 million de dollars en une journée.""",
            "category": "risk"
        }
    ]
    
    return documents


if __name__ == "__main__":
    # Test RAG
    print("🔧 Testing RAG Engine\n")
    
    # Initialize
    rag = SimpleRAG()
    
    # Add sample documents
    print("📚 Adding sample documents...")
    for doc in create_sample_documents():
        rag.add_document(doc['content'], {
            'title': doc['title'],
            'category': doc['category']
        }, save=False)
    
    rag.save_index()
    
    # Get stats
    stats = rag.get_stats()
    print(f"\n📊 Index Stats:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Categories: {stats['categories']}")
    
    # Test search
    print("\n🔍 Testing search...")
    queries = [
        "Qu'est-ce que le ratio de Sharpe?",
        "Comment diversifier un portfolio?",
        "Expliquer la VaR"
    ]
    
    for query in queries:
        print(f"\n  Query: {query}")
        results = rag.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['metadata']['title']} (score: {result['score']:.3f})")
            print(f"       {result['text'][:100]}...")
    
    print("\n✅ RAG Engine test complete!")
