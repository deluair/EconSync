#!/usr/bin/env python3
"""
Command-line interface for EconSync.
"""

import typer
from typing import Optional, List
from pathlib import Path
import json

from .core.config import EconSyncConfig
from .core.agent import EconSyncAgent
from .utils.logger import setup_logger

app = typer.Typer(help="EconSync: Smart Agent for Applied Economics Research")


@app.command()
def init(
    config_path: str = typer.Option("configs/default.yaml", help="Configuration file path"),
    data_dir: str = typer.Option("data", help="Data directory"),
    force: bool = typer.Option(False, help="Force initialization even if already exists")
):
    """Initialize EconSync project."""
    typer.echo("🚀 Initializing EconSync...")
    
    # Create directories
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(".cache").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Load or create config
    if Path(config_path).exists() and not force:
        typer.echo(f"✅ Configuration already exists at {config_path}")
        config = EconSyncConfig.from_yaml(config_path)
    else:
        config = EconSyncConfig()
        config.save_yaml(config_path)
        typer.echo(f"✅ Configuration created at {config_path}")
    
    # Initialize agent
    try:
        agent = EconSyncAgent(config)
        typer.echo("✅ EconSync agent initialized successfully!")
        
        # Display status
        status = agent.get_status()
        typer.echo(f"📊 Version: {status['version']}")
        typer.echo(f"🔍 RAG documents: {status['rag_retriever']['collection_stats']['total_documents']}")
        
    except Exception as e:
        typer.echo(f"❌ Initialization failed: {e}")
        raise typer.Exit(1)


@app.command()
def analyze(
    query: str = typer.Argument(..., help="Economic analysis query"),
    config_path: str = typer.Option("configs/default.yaml", help="Configuration file path"),
    adapters: Optional[List[str]] = typer.Option(None, help="LoRA adapters to use"),
    output: Optional[str] = typer.Option(None, help="Output file for results"),
    verbose: bool = typer.Option(False, help="Verbose output")
):
    """Perform economic analysis."""
    typer.echo(f"🔍 Analyzing: {query}")
    
    # Load configuration
    if Path(config_path).exists():
        config = EconSyncConfig.from_yaml(config_path)
    else:
        config = EconSyncConfig()
    
    if verbose:
        config.debug = True
    
    # Initialize agent
    agent = EconSyncAgent(config)
    
    try:
        # Perform analysis
        result = agent.analyze(query, use_adapters=adapters)
        
        typer.echo("✅ Analysis completed!")
        typer.echo(f"🧠 Reasoning steps: {len(result['reasoning']['reasoning_steps'])}")
        typer.echo(f"📊 Insights: {len(result['analysis']['insights'])}")
        typer.echo(f"🎯 Confidence: {result['analysis']['confidence_score']:.3f}")
        
        # Display insights
        typer.echo("\n💡 Key Insights:")
        for i, insight in enumerate(result['analysis']['insights'][:5], 1):
            typer.echo(f"  {i}. {insight}")
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            typer.echo(f"💾 Results saved to {output}")
    
    except Exception as e:
        typer.echo(f"❌ Analysis failed: {e}")
        raise typer.Exit(1)


@app.command()
def forecast(
    variable: str = typer.Argument(..., help="Variable to forecast"),
    horizon: int = typer.Option(12, help="Forecast horizon"),
    method: str = typer.Option("auto", help="Forecasting method"),
    config_path: str = typer.Option("configs/default.yaml", help="Configuration file path"),
    output: Optional[str] = typer.Option(None, help="Output file for forecast")
):
    """Generate economic forecasts."""
    typer.echo(f"📈 Forecasting {variable} for {horizon} periods...")
    
    # Load configuration
    if Path(config_path).exists():
        config = EconSyncConfig.from_yaml(config_path)
    else:
        config = EconSyncConfig()
    
    # Initialize agent
    agent = EconSyncAgent(config)
    
    try:
        # Generate forecast
        result = agent.forecast(variable=variable, horizon=horizon, method=method)
        
        typer.echo("✅ Forecast completed!")
        typer.echo(f"📊 Method: {result['method_used']}")
        typer.echo(f"🎯 Forecast values: {result['forecast'][:5]}...")  # Show first 5
        
        # Save forecast if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            typer.echo(f"💾 Forecast saved to {output}")
    
    except Exception as e:
        typer.echo(f"❌ Forecast failed: {e}")
        raise typer.Exit(1)


@app.command()
def policy(
    description: str = typer.Argument(..., help="Policy description"),
    sectors: Optional[List[str]] = typer.Option(None, help="Affected sectors"),
    config_path: str = typer.Option("configs/default.yaml", help="Configuration file path"),
    output: Optional[str] = typer.Option(None, help="Output file for analysis")
):
    """Analyze policy impacts."""
    typer.echo(f"🏛️ Analyzing policy: {description}")
    
    # Load configuration
    if Path(config_path).exists():
        config = EconSyncConfig.from_yaml(config_path)
    else:
        config = EconSyncConfig()
    
    # Initialize agent
    agent = EconSyncAgent(config)
    
    try:
        # Analyze policy
        result = agent.policy_impact(description, sectors)
        
        typer.echo("✅ Policy analysis completed!")
        typer.echo(f"🧠 Reasoning steps: {len(result['reasoning_steps'])}")
        typer.echo(f"📋 Conclusion: {result['conclusion'][:100]}...")
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            typer.echo(f"💾 Analysis saved to {output}")
    
    except Exception as e:
        typer.echo(f"❌ Policy analysis failed: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    config_path: str = typer.Option("configs/default.yaml", help="Configuration file path")
):
    """Show EconSync agent status."""
    # Load configuration
    if Path(config_path).exists():
        config = EconSyncConfig.from_yaml(config_path)
    else:
        config = EconSyncConfig()
    
    try:
        # Initialize agent
        agent = EconSyncAgent(config)
        
        # Get status
        status = agent.get_status()
        
        typer.echo("🤖 EconSync Agent Status")
        typer.echo("=" * 30)
        typer.echo(f"📊 Version: {status['version']}")
        typer.echo(f"🧠 Active adapters: {status['active_adapters']}")
        typer.echo(f"💾 Loaded datasets: {status['data_manager']['loaded_datasets']}")
        typer.echo(f"🔍 RAG documents: {status['rag_retriever']['collection_stats']['total_documents']}")
        typer.echo(f"📂 Data directory: {status['data_manager']['data_directory']}")
        typer.echo(f"💾 Memory usage: {status['data_manager']['total_memory_usage']:,.0f} bytes")
    
    except Exception as e:
        typer.echo(f"❌ Status check failed: {e}")
        raise typer.Exit(1)


@app.command()
def generate_data(
    datasets: Optional[List[str]] = typer.Option(
        ["macroeconomic_indicators", "financial_markets", "trade_data"],
        help="Datasets to generate"
    ),
    output_dir: str = typer.Option("data", help="Output directory"),
    force: bool = typer.Option(False, help="Force regeneration")
):
    """Generate synthetic economic datasets."""
    typer.echo("📊 Generating synthetic economic datasets...")
    
    # Import here to avoid circular imports
    from .data.generators import DataGenerator
    
    config = EconSyncConfig()
    generator = DataGenerator(config)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dataset_name in datasets:
        typer.echo(f"Generating {dataset_name}...")
        
        # Check if exists
        data_file = output_path / f"{dataset_name}.parquet"
        if data_file.exists() and not force:
            typer.echo(f"⚠️ {dataset_name} already exists, skipping")
            continue
        
        try:
            # Generate dataset
            if dataset_name == "macroeconomic_indicators":
                result = generator.generate_macroeconomic_data()
            elif dataset_name == "financial_markets":
                result = generator.generate_financial_data()
            elif dataset_name == "trade_data":
                result = generator.generate_trade_data()
            elif dataset_name == "firm_level":
                result = generator.generate_firm_data()
            elif dataset_name == "policy_documents":
                result = generator.generate_policy_data()
            else:
                typer.echo(f"❌ Unknown dataset: {dataset_name}")
                continue
            
            # Save data
            result["data"].to_parquet(data_file, index=False)
            
            # Save metadata
            metadata_file = output_path / f"{dataset_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(result["metadata"], f, indent=2, default=str)
            
            typer.echo(f"✅ {dataset_name}: {result['data'].shape[0]} rows generated")
        
        except Exception as e:
            typer.echo(f"❌ Failed to generate {dataset_name}: {e}")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main() 