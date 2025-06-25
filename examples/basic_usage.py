#!/usr/bin/env python3
"""
Basic usage example for EconSync.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import econsync
sys.path.insert(0, str(Path(__file__).parent.parent))

from econsync import EconSyncAgent, EconSyncConfig


def main():
    print("🔬 EconSync Basic Usage Example")
    print("=" * 50)
    
    # Initialize the EconSync agent
    print("\n1. Initializing EconSync Agent...")
    config = EconSyncConfig()
    agent = EconSyncAgent(config)
    print("✅ Agent initialized successfully!")
    
    # Load economic data
    print("\n2. Loading macroeconomic data...")
    try:
        data_result = agent.load_data("macroeconomic_indicators")
        print(f"✅ Data loaded: {data_result['data'].shape[0]} observations")
        print(f"   Variables: {list(data_result['data'].columns)}")
    except Exception as e:
        print(f"⚠️ Data loading: {e}")
    
    # Perform economic analysis
    print("\n3. Performing economic analysis...")
    queries = [
        "What are the inflation trends for 2024?",
        "How does unemployment relate to GDP growth?",
        "What is the relationship between interest rates and inflation?",
        "Forecast GDP growth for the next 12 months"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query}")
        try:
            result = agent.analyze(query)
            
            print(f"   ✅ Analysis completed in {len(result['reasoning']['reasoning_steps'])} reasoning steps")
            print(f"   📊 Insights found: {len(result['analysis']['insights'])}")
            
            # Display key insights
            for insight in result['analysis']['insights'][:2]:  # Show first 2 insights
                print(f"      • {insight}")
                
        except Exception as e:
            print(f"   ⚠️ Analysis failed: {e}")
    
    # Generate forecasts
    print("\n4. Generating economic forecasts...")
    forecast_variables = ["gdp_growth", "inflation_cpi", "unemployment_rate"]
    
    for variable in forecast_variables:
        try:
            forecast_result = agent.forecast(variable, horizon=6)
            print(f"   📈 {variable}: {len(forecast_result['forecast'])} periods forecasted")
        except Exception as e:
            print(f"   ⚠️ Forecast for {variable}: {e}")
    
    # Policy impact analysis
    print("\n5. Analyzing policy impacts...")
    policy_examples = [
        "Increase federal minimum wage to $15 per hour",
        "Implement carbon tax of $50 per ton",
        "Reduce corporate tax rate from 21% to 18%"
    ]
    
    for policy in policy_examples:
        try:
            policy_result = agent.policy_impact(policy)
            print(f"   🏛️ Policy analyzed: {policy[:40]}...")
            print(f"      Reasoning steps: {len(policy_result['reasoning_steps'])}")
        except Exception as e:
            print(f"   ⚠️ Policy analysis failed: {e}")
    
    # Show agent status
    print("\n6. Agent Status:")
    status = agent.get_status()
    print(f"   📊 Active adapters: {status['active_adapters']}")
    print(f"   💾 Loaded datasets: {status['data_manager']['loaded_datasets']}")
    print(f"   🔍 RAG documents: {status['rag_retriever']['collection_stats']['total_documents']}")
    
    print("\n🎉 Basic usage example completed!")
    print("\nNext steps:")
    print("• Explore advanced features in the notebooks/ directory")
    print("• Add your own economic data and literature")
    print("• Fine-tune LoRA adapters for specific domains")
    print("• Customize the ReAct reasoning framework")


if __name__ == "__main__":
    main() 