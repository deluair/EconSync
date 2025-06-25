"""
Main EconSync Agent that orchestrates LoRA, RAG, and ReAct components.
"""

from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

from .config import EconSyncConfig
from ..adapters.manager import AdapterManager
from ..rag.retriever import RAGRetriever
from ..react.framework import ReActFramework
from ..data.manager import DataManager
from ..models.economic import EconomicModel
from ..analytics.analyzer import EconomicAnalyzer
from ..utils.logger import setup_logger


class EconSyncAgent:
    """
    Main agent that integrates LoRA adapters, RAG retrieval, and ReAct framework
    for comprehensive economic analysis and research.
    """
    
    def __init__(self, config: Optional[EconSyncConfig] = None):
        """
        Initialize the EconSync agent.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or EconSyncConfig()
        self.logger = setup_logger("EconSyncAgent", debug=self.config.debug)
        
        # Initialize core components
        self._initialize_components()
        
        self.logger.info("EconSync Agent initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Data management
            self.data_manager = DataManager(self.config)
            
            # LoRA adapter management
            self.adapter_manager = AdapterManager(self.config)
            
            # RAG retrieval system
            self.rag_retriever = RAGRetriever(self.config)
            
            # ReAct reasoning framework
            self.react_framework = ReActFramework(self.config)
            
            # Economic models
            self.economic_model = EconomicModel(self.config)
            
            # Analytics and analysis
            self.analyzer = EconomicAnalyzer(self.config)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def load_data(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """
        Load economic dataset for analysis.
        
        Args:
            dataset_name: Name of the dataset to load
            **kwargs: Additional parameters for data loading
            
        Returns:
            Dictionary containing loaded data and metadata
        """
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            data = self.data_manager.load_dataset(dataset_name, **kwargs)
            self.logger.info(f"Successfully loaded {dataset_name}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def analyze(self, 
                query: str, 
                data_sources: Optional[List[str]] = None,
                use_adapters: Optional[List[str]] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive economic analysis for the given query.
        
        Args:
            query: Research question or analysis request
            data_sources: List of data sources to use
            use_adapters: List of LoRA adapters to activate
            context: Additional context for the analysis
            
        Returns:
            Analysis results including insights, data, and visualizations
        """
        self.logger.info(f"Starting analysis for query: {query}")
        
        try:
            # Step 1: Use ReAct framework to decompose the query
            reasoning_result = self.react_framework.reason_and_act(
                query=query,
                context=context or {}
            )
            
            # Step 2: Retrieve relevant information using RAG
            relevant_docs = self.rag_retriever.retrieve(
                query=query,
                top_k=self.config.rag.top_k
            )
            
            # Step 3: Activate appropriate LoRA adapters
            if use_adapters:
                self.adapter_manager.activate_adapters(use_adapters)
            else:
                # Auto-select adapters based on query
                suggested_adapters = self._suggest_adapters(query)
                self.adapter_manager.activate_adapters(suggested_adapters)
            
            # Step 4: Perform economic analysis
            analysis_result = self.analyzer.analyze(
                query=query,
                reasoning=reasoning_result,
                retrieved_docs=relevant_docs,
                data_sources=data_sources
            )
            
            # Step 5: Generate comprehensive response
            response = {
                "query": query,
                "reasoning": reasoning_result,
                "retrieved_knowledge": relevant_docs,
                "analysis": analysis_result,
                "timestamp": self.data_manager.get_current_timestamp(),
                "config_used": {
                    "adapters": self.adapter_manager.get_active_adapters(),
                    "model": self.config.model.base_model,
                    "rag_model": self.config.rag.embedding_model
                }
            }
            
            self.logger.info("Analysis completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def _suggest_adapters(self, query: str) -> List[str]:
        """
        Suggest appropriate LoRA adapters based on the query content.
        
        Args:
            query: The analysis query
            
        Returns:
            List of suggested adapter names
        """
        # Simple keyword-based suggestion (can be enhanced with ML)
        query_lower = query.lower()
        suggested = []
        
        if any(word in query_lower for word in ["trade", "export", "import", "tariff"]):
            suggested.append("trade_economics")
        
        if any(word in query_lower for word in ["market", "stock", "bond", "financial"]):
            suggested.append("financial_markets")
        
        if any(word in query_lower for word in ["agricultural", "commodity", "farming", "food"]):
            suggested.append("agricultural_economics")
        
        if any(word in query_lower for word in ["policy", "regulation", "government"]):
            suggested.append("policy_analysis")
        
        if any(word in query_lower for word in ["gdp", "inflation", "unemployment", "macro"]):
            suggested.append("macroeconomic_forecasting")
        
        # Default to general economic analysis if no specific matches
        if not suggested:
            suggested = ["macroeconomic_forecasting"]
        
        return suggested
    
    def forecast(self, 
                 variable: str, 
                 horizon: int, 
                 method: str = "auto") -> Dict[str, Any]:
        """
        Generate economic forecasts for specific variables.
        
        Args:
            variable: Economic variable to forecast (e.g., "GDP", "inflation")
            horizon: Forecast horizon in periods
            method: Forecasting method ("auto", "arima", "var", "ml")
            
        Returns:
            Forecast results with confidence intervals
        """
        self.logger.info(f"Generating forecast for {variable}, horizon: {horizon}")
        
        try:
            # Activate forecasting adapter
            self.adapter_manager.activate_adapters(["macroeconomic_forecasting"])
            
            # Generate forecast using economic model
            forecast_result = self.economic_model.forecast(
                variable=variable,
                horizon=horizon,
                method=method
            )
            
            return forecast_result
            
        except Exception as e:
            self.logger.error(f"Forecasting failed: {e}")
            raise
    
    def policy_impact(self, 
                     policy_description: str, 
                     affected_sectors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze the potential impact of economic policies.
        
        Args:
            policy_description: Description of the proposed policy
            affected_sectors: List of sectors that might be affected
            
        Returns:
            Policy impact analysis results
        """
        self.logger.info(f"Analyzing policy impact: {policy_description}")
        
        try:
            # Activate policy analysis adapter
            self.adapter_manager.activate_adapters(["policy_analysis"])
            
            # Use ReAct framework for policy analysis
            policy_analysis = self.react_framework.analyze_policy_impact(
                policy_description=policy_description,
                affected_sectors=affected_sectors
            )
            
            return policy_analysis
            
        except Exception as e:
            self.logger.error(f"Policy impact analysis failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the EconSync agent.
        
        Returns:
            Status information for all components
        """
        return {
            "agent_status": "active",
            "config": self.config.to_dict(),
            "active_adapters": self.adapter_manager.get_active_adapters(),
            "data_manager": self.data_manager.get_status(),
            "rag_retriever": self.rag_retriever.get_status(),
            "version": self.config.version
        }
    
    def save_session(self, session_path: str):
        """Save current session state."""
        self.logger.info(f"Saving session to {session_path}")
        # Implementation for saving session state
        pass
    
    def load_session(self, session_path: str):
        """Load previous session state."""
        self.logger.info(f"Loading session from {session_path}")
        # Implementation for loading session state
        pass 