#!/usr/bin/env python3
"""
Script to generate synthetic economic datasets for EconSync.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import econsync
sys.path.insert(0, str(Path(__file__).parent.parent))

from econsync import EconSyncConfig, DataGenerator
from econsync.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic economic datasets")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Configuration file path")
    parser.add_argument("--datasets", nargs="+", 
                       default=["macroeconomic_indicators", "financial_markets", 
                               "trade_data", "firm_level", "policy_documents"],
                       help="Datasets to generate")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for datasets")
    parser.add_argument("--force", action="store_true",
                       help="Force regeneration of existing datasets")
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = EconSyncConfig.from_yaml(args.config)
    else:
        print(f"Configuration file {args.config} not found, using defaults")
        config = EconSyncConfig()
    
    # Setup logging
    logger = setup_logger("DatasetGenerator", debug=config.debug)
    logger.info("Starting dataset generation")
    
    # Initialize data generator
    generator = DataGenerator(config)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate each dataset
    for dataset_name in args.datasets:
        logger.info(f"Generating dataset: {dataset_name}")
        
        try:
            # Check if dataset already exists
            data_file = output_dir / f"{dataset_name}.parquet"
            if data_file.exists() and not args.force:
                logger.info(f"Dataset {dataset_name} already exists, skipping (use --force to regenerate)")
                continue
            
            # Generate dataset based on type
            if dataset_name == "macroeconomic_indicators":
                result = generator.generate_macroeconomic_data()
            elif dataset_name == "financial_markets":
                result = generator.generate_financial_data(n_assets=500)
            elif dataset_name == "trade_data":
                result = generator.generate_trade_data(n_countries=50)
            elif dataset_name == "firm_level":
                result = generator.generate_firm_data(n_firms=50000)
            elif dataset_name == "policy_documents":
                result = generator.generate_policy_data(n_policies=10000)
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            # Save dataset
            data = result["data"]
            metadata = result["metadata"]
            
            # Save data as parquet
            data.to_parquet(data_file, index=False)
            
            # Save metadata as JSON
            metadata_file = output_dir / f"{dataset_name}_metadata.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Successfully generated {dataset_name}: {data.shape[0]} rows, {data.shape[1]} columns")
            
        except Exception as e:
            logger.error(f"Failed to generate {dataset_name}: {e}")
    
    logger.info("Dataset generation completed")


if __name__ == "__main__":
    main() 