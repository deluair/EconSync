"""
Tests for data generators.
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path to import econsync
sys.path.insert(0, str(Path(__file__).parent.parent))

from econsync import EconSyncConfig, DataGenerator


class TestDataGenerator:
    """Test suite for DataGenerator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = EconSyncConfig()
        self.generator = DataGenerator(self.config)
        
    def test_macroeconomic_data_generation(self):
        """Test macroeconomic data generation."""
        result = self.generator.generate_macroeconomic_data()
        
        assert "data" in result
        assert "metadata" in result
        
        data = result["data"]
        assert len(data) > 0
        assert "date" in data.columns
        assert "gdp_growth" in data.columns
        assert "inflation_cpi" in data.columns
        assert "unemployment_rate" in data.columns
        
        # Check data types
        assert data["gdp_growth"].dtype in ["float64", "float32"]
        assert data["inflation_cpi"].dtype in ["float64", "float32"]
        
    def test_financial_data_generation(self):
        """Test financial data generation."""
        result = self.generator.generate_financial_data(n_assets=100)
        
        assert "data" in result
        assert "metadata" in result
        
        data = result["data"]
        assert len(data) > 0
        assert "asset_id" in data.columns
        assert "asset_type" in data.columns
        assert "price" in data.columns
        assert "volume" in data.columns
        
        # Check unique assets
        unique_assets = data["asset_id"].nunique()
        assert unique_assets <= 100
        
    def test_trade_data_generation(self):
        """Test trade data generation."""
        result = self.generator.generate_trade_data(n_countries=10)
        
        assert "data" in result
        assert "metadata" in result
        
        data = result["data"]
        assert len(data) > 0
        assert "exporter" in data.columns
        assert "importer" in data.columns
        assert "trade_value_usd" in data.columns
        assert "product_category" in data.columns
        
        # Check unique countries
        unique_countries = set(data["exporter"].unique()) | set(data["importer"].unique())
        assert len(unique_countries) <= 10
        
    def test_firm_data_generation(self):
        """Test firm-level data generation."""
        result = self.generator.generate_firm_data(n_firms=1000)
        
        assert "data" in result
        assert "metadata" in result
        
        data = result["data"]
        assert len(data) == 1000
        assert "firm_id" in data.columns
        assert "industry_code" in data.columns
        assert "employment" in data.columns
        assert "revenue" in data.columns
        
        # Check data validity
        assert (data["employment"] > 0).all()
        assert (data["revenue"] >= 0).all()
        
    def test_policy_data_generation(self):
        """Test policy data generation."""
        result = self.generator.generate_policy_data(n_policies=100)
        
        assert "data" in result
        assert "metadata" in result
        
        data = result["data"]
        assert len(data) == 100
        assert "policy_id" in data.columns
        assert "policy_type" in data.columns
        assert "policy_date" in data.columns
        assert "affected_sectors" in data.columns
        
        # Check policy types
        policy_types = data["policy_type"].unique()
        expected_types = ['Trade', 'Monetary', 'Fiscal', 'Environmental', 'Labor', 'Healthcare', 'Education']
        assert all(pt in expected_types for pt in policy_types)
        
    def test_data_consistency(self):
        """Test data consistency across generations."""
        # Generate same data twice with same seed
        self.config.seed = 42
        generator1 = DataGenerator(self.config)
        generator2 = DataGenerator(self.config)
        
        result1 = generator1.generate_macroeconomic_data()
        result2 = generator2.generate_macroeconomic_data()
        
        # Results should be identical with same seed
        assert result1["data"].equals(result2["data"])
        
    def test_metadata_completeness(self):
        """Test metadata completeness."""
        result = self.generator.generate_macroeconomic_data()
        metadata = result["metadata"]
        
        required_fields = ["dataset_name", "description", "n_observations", "variables"]
        for field in required_fields:
            assert field in metadata
            
        assert metadata["n_observations"] > 0
        assert len(metadata["variables"]) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 