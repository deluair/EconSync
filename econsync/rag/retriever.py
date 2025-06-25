"""
RAG Retrieval System for EconSync.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime

from ..core.config import EconSyncConfig
from ..utils.logger import setup_logger


class RAGRetriever:
    """
    Retrieval-Augmented Generation system for economic knowledge.
    """
    
    def __init__(self, config: EconSyncConfig):
        """
        Initialize the RAG retriever.
        
        Args:
            config: EconSync configuration
        """
        self.config = config
        self.logger = setup_logger("RAGRetriever", debug=config.debug)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.rag.embedding_model)
        
        # Initialize vector database
        self.client = chromadb.PersistentClient(path=str(Path(config.data.cache_dir) / "chroma"))
        
        try:
            self.collection = self.client.get_collection(config.rag.collection_name)
            self.logger.info(f"Loaded existing collection: {config.rag.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=config.rag.collection_name,
                metadata={"description": "EconSync knowledge base"}
            )
            self.logger.info(f"Created new collection: {config.rag.collection_name}")
            # Initialize with sample economic knowledge
            self._initialize_knowledge_base()
        
        self.logger.info("RAGRetriever initialized successfully")
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with sample economic content."""
        self.logger.info("Initializing knowledge base with sample economic content")
        
        # Sample economic knowledge
        sample_docs = [
            {
                "id": "macro_001",
                "text": "Gross Domestic Product (GDP) is the total monetary value of all finished goods and services produced within a country's borders in a specific time period. It serves as a broad measure of overall domestic production and economic health.",
                "source": "Economic Fundamentals",
                "category": "Macroeconomics",
                "tags": ["GDP", "economic indicators", "national accounts"]
            },
            {
                "id": "macro_002", 
                "text": "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power. Central banks attempt to limit inflation through monetary policy to maintain price stability.",
                "source": "Monetary Economics",
                "category": "Macroeconomics",
                "tags": ["inflation", "monetary policy", "price level"]
            },
            {
                "id": "trade_001",
                "text": "Comparative advantage is an economic theory that explains how countries benefit from international trade by specializing in producing goods where they have the lowest opportunity cost, even if they don't have an absolute advantage.",
                "source": "International Trade Theory",
                "category": "International Economics", 
                "tags": ["comparative advantage", "trade theory", "specialization"]
            },
            {
                "id": "finance_001",
                "text": "The Efficient Market Hypothesis (EMH) states that asset prices fully reflect all available information, making it impossible to consistently achieve returns in excess of average market returns on a risk-adjusted basis.",
                "source": "Financial Economics",
                "category": "Finance",
                "tags": ["EMH", "market efficiency", "asset pricing"]
            },
            {
                "id": "policy_001",
                "text": "Fiscal policy refers to government revenue collection and expenditure to influence the economy. During recessions, expansionary fiscal policy through increased spending or tax cuts can stimulate economic growth.",
                "source": "Public Economics",
                "category": "Policy",
                "tags": ["fiscal policy", "government spending", "economic stimulus"]
            },
            {
                "id": "labor_001",
                "text": "The Phillips Curve illustrates the inverse relationship between unemployment and inflation rates in the short run. However, this relationship may break down in the long run due to adaptive expectations.",
                "source": "Labor Economics",
                "category": "Labor",
                "tags": ["Phillips curve", "unemployment", "inflation", "trade-off"]
            },
            {
                "id": "micro_001",
                "text": "Price elasticity of demand measures the responsiveness of quantity demanded to changes in price. Elastic demand (elasticity > 1) indicates consumers are highly responsive to price changes.",
                "source": "Microeconomics",
                "category": "Microeconomics",
                "tags": ["elasticity", "demand", "price sensitivity"]
            },
            {
                "id": "development_001",
                "text": "The Solow growth model explains long-run economic growth through capital accumulation, labor force growth, and technological progress. It predicts convergence of per capita income levels across countries.",
                "source": "Development Economics",
                "category": "Growth",
                "tags": ["Solow model", "economic growth", "convergence", "capital"]
            }
        ]
        
        # Add documents to collection
        texts = [doc["text"] for doc in sample_docs]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{k: v for k, v in doc.items() if k != "text"} for doc in sample_docs],
            ids=[doc["id"] for doc in sample_docs]
        )
        
        self.logger.info(f"Added {len(sample_docs)} sample documents to knowledge base")
    
    def add_documents(self, 
                     documents: List[Dict[str, Any]],
                     batch_size: int = 100) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents with 'text' and metadata
            batch_size: Batch size for processing
        """
        self.logger.info(f"Adding {len(documents)} documents to knowledge base")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            texts = [doc["text"] for doc in batch]
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Generate IDs if not provided
            ids = [doc.get("id", f"doc_{i + j}") for j, doc in enumerate(batch)]
            
            # Extract metadata
            metadatas = []
            for doc in batch:
                metadata = {k: v for k, v in doc.items() if k not in ["text", "id"]}
                metadata["added_at"] = datetime.now().isoformat()
                metadatas.append(metadata)
            
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        
        self.logger.info(f"Successfully added {len(documents)} documents")
    
    def retrieve(self, 
                query: str, 
                top_k: Optional[int] = None,
                filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_criteria: Metadata filters
            
        Returns:
            List of relevant documents with scores
        """
        if top_k is None:
            top_k = self.config.rag.top_k
        
        self.logger.debug(f"Retrieving documents for query: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_criteria
        )
        
        # Format results
        retrieved_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = {
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "metadata": results['metadatas'][0][i] if results['metadatas'][0] else {}
                }
                
                # Only include if above similarity threshold
                if doc["score"] >= self.config.rag.similarity_threshold:
                    retrieved_docs.append(doc)
        
        self.logger.debug(f"Retrieved {len(retrieved_docs)} relevant documents")
        return retrieved_docs
    
    def semantic_search(self, 
                       query: str,
                       category: Optional[str] = None,
                       top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional category filtering.
        
        Args:
            query: Search query
            category: Optional category filter
            top_k: Number of results
            
        Returns:
            Search results
        """
        filter_criteria = None
        if category:
            filter_criteria = {"category": category}
        
        return self.retrieve(query, top_k=top_k, filter_criteria=filter_criteria)
    
    def get_related_concepts(self, 
                           concept: str,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get concepts related to the input concept.
        
        Args:
            concept: Input concept
            top_k: Number of related concepts
            
        Returns:
            Related concepts
        """
        query = f"Economic concepts related to {concept}"
        return self.retrieve(query, top_k=top_k)
    
    def add_economic_literature(self, 
                              papers: List[Dict[str, Any]]) -> None:
        """
        Add economic research papers to the knowledge base.
        
        Args:
            papers: List of paper dictionaries with title, abstract, etc.
        """
        self.logger.info(f"Adding {len(papers)} economic papers")
        
        documents = []
        for paper in papers:
            # Combine title and abstract for better retrieval
            text = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}"
            
            doc = {
                "text": text,
                "id": paper.get("id", f"paper_{len(documents)}"),
                "source": "Academic Literature",
                "category": "Research",
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "journal": paper.get("journal", ""),
                "year": paper.get("year", ""),
                "doi": paper.get("doi", ""),
                "keywords": paper.get("keywords", [])
            }
            documents.append(doc)
        
        self.add_documents(documents)
    
    def update_document(self, 
                       doc_id: str, 
                       new_text: str,
                       new_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            new_text: New document text
            new_metadata: New metadata
        """
        self.logger.info(f"Updating document: {doc_id}")
        
        # Generate new embedding
        new_embedding = self.embedding_model.encode([new_text]).tolist()[0]
        
        # Update metadata
        metadata = new_metadata or {}
        metadata["updated_at"] = datetime.now().isoformat()
        
        # Update in collection
        self.collection.update(
            ids=[doc_id],
            documents=[new_text],
            embeddings=[new_embedding],
            metadatas=[metadata]
        )
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document from the knowledge base.
        
        Args:
            doc_id: Document ID to delete
        """
        self.logger.info(f"Deleting document: {doc_id}")
        self.collection.delete(ids=[doc_id])
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Collection statistics
        """
        count = self.collection.count()
        
        # Get sample of metadata to understand categories
        sample_results = self.collection.peek(limit=min(100, count))
        categories = set()
        sources = set()
        
        if sample_results['metadatas']:
            for metadata in sample_results['metadatas']:
                if 'category' in metadata:
                    categories.add(metadata['category'])
                if 'source' in metadata:
                    sources.add(metadata['source'])
        
        return {
            "total_documents": count,
            "categories": list(categories),
            "sources": list(sources),
            "embedding_model": self.config.rag.embedding_model,
            "collection_name": self.config.rag.collection_name
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the RAG retriever.
        
        Returns:
            Status information
        """
        return {
            "status": "active",
            "collection_stats": self.get_collection_stats(),
            "config": {
                "embedding_model": self.config.rag.embedding_model,
                "top_k": self.config.rag.top_k,
                "similarity_threshold": self.config.rag.similarity_threshold
            }
        } 