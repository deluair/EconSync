"""
Tests for the EconSync agent.
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path to import econsync
sys.path.insert(0, str(Path(__file__).parent.parent))

from econsync import EconSyncAgent, EconSyncConfig


class TestEconSyncAgent:
    """Test suite for EconSyncAgent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = EconSyncConfig()
        self.config.debug = True
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = EconSyncAgent(self.config)
        assert agent is not None
        assert agent.config == self.config
        
    def test_agent_status(self):
        """Test agent status reporting."""
        agent = EconSyncAgent(self.config)
        status = agent.get_status()
        
        assert "agent_status" in status
        assert "version" in status
        assert "config" in status
        assert status["agent_status"] == "active"
        
    def test_data_loading(self):
        """Test data loading functionality."""
        agent = EconSyncAgent(self.config)
        
        # Test loading macroeconomic data
        result = agent.load_data("macroeconomic_indicators")
        
        assert "data" in result
        assert "metadata" in result
        assert result["data"] is not None
        assert len(result["data"]) > 0
        
    def test_basic_analysis(self):
        """Test basic economic analysis."""
        agent = EconSyncAgent(self.config)
        
        query = "What are the inflation trends?"
        result = agent.analyze(query)
        
        assert "query" in result
        assert "reasoning" in result
        assert "analysis" in result
        assert result["query"] == query
        
    def test_forecasting(self):
        """Test economic forecasting."""
        agent = EconSyncAgent(self.config)
        
        result = agent.forecast("gdp_growth", horizon=6)
        
        assert "forecast" in result
        assert "method_used" in result
        assert len(result["forecast"]) == 6
        
    def test_policy_analysis(self):
        """Test policy impact analysis."""
        agent = EconSyncAgent(self.config)
        
        policy = "Increase minimum wage to $15"
        result = agent.policy_impact(policy)
        
        assert "reasoning_steps" in result
        assert "conclusion" in result
        
    def test_adapter_management(self):
        """Test LoRA adapter management."""
        agent = EconSyncAgent(self.config)
        
        # Test getting available adapters
        available = agent.adapter_manager.get_available_adapters()
        assert len(available) > 0
        assert "financial_markets" in available
        
        # Test activating adapters
        agent.adapter_manager.activate_adapters(["financial_markets"])
        active = agent.adapter_manager.get_active_adapters()
        assert "financial_markets" in active


class TestEconSyncConfig:
    """Test suite for EconSyncConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EconSyncConfig()
        
        assert config.project_name == "EconSync"
        assert config.version == "0.1.0"
        assert config.seed == 42
        
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = EconSyncConfig()
        config_dict = config.to_dict()
        
        assert "project_name" in config_dict
        assert "version" in config_dict
        assert "lora" in config_dict
        assert "rag" in config_dict
        
    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {
            "project_name": "TestProject",
            "version": "0.2.0",
            "lora": {"rank": 32},
            "rag": {"top_k": 10}
        }
        
        config = EconSyncConfig.from_dict(config_dict)
        
        assert config.project_name == "TestProject"
        assert config.version == "0.2.0"
        assert config.lora.rank == 32
        assert config.rag.top_k == 10


if __name__ == "__main__":
    pytest.main([__file__]) 