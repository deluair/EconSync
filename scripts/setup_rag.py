#!/usr/bin/env python3
"""
Script to initialize the RAG knowledge base for EconSync.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import econsync
sys.path.insert(0, str(Path(__file__).parent.parent))

from econsync import EconSyncConfig
from econsync.rag.retriever import RAGRetriever
from econsync.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Initialize RAG knowledge base")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Configuration file path")
    parser.add_argument("--reset", action="store_true",
                       help="Reset the existing knowledge base")
    parser.add_argument("--add-sample-docs", action="store_true", default=True,
                       help="Add sample economic documents")
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = EconSyncConfig.from_yaml(args.config)
    else:
        print(f"Configuration file {args.config} not found, using defaults")
        config = EconSyncConfig()
    
    # Setup logging
    logger = setup_logger("RAGSetup", debug=config.debug)
    logger.info("Initializing RAG knowledge base")
    
    # Initialize RAG retriever (this will create the collection if it doesn't exist)
    retriever = RAGRetriever(config)
    
    if args.reset:
        logger.info("Resetting knowledge base")
        # Delete and recreate collection
        retriever.client.delete_collection(config.rag.collection_name)
        retriever.collection = retriever.client.create_collection(
            name=config.rag.collection_name,
            metadata={"description": "EconSync knowledge base"}
        )
        retriever._initialize_knowledge_base()
    
    if args.add_sample_docs:
        logger.info("Adding sample economic literature")
        
        # Sample economic papers and documents
        sample_papers = [
            {
                "title": "The Great Recession and the Role of Monetary Policy",
                "abstract": "This paper examines the causes and consequences of the 2008 financial crisis, focusing on the role of monetary policy in both the buildup to the crisis and the recovery efforts. We analyze the effectiveness of unconventional monetary policies including quantitative easing and forward guidance.",
                "authors": "Smith, J. and Johnson, K.",
                "journal": "American Economic Review",
                "year": "2020",
                "doi": "10.1257/aer.20201234",
                "keywords": ["monetary policy", "financial crisis", "quantitative easing", "recession"]
            },
            {
                "title": "International Trade and Economic Growth in Developing Countries",
                "abstract": "We investigate the relationship between international trade openness and economic growth using panel data from 80 developing countries over the period 1990-2020. Our findings suggest that trade liberalization has heterogeneous effects depending on institutional quality and initial development levels.",
                "authors": "Garcia, M., Lee, S., and Patel, R.",
                "journal": "Journal of Development Economics",
                "year": "2022",
                "doi": "10.1016/j.jdeveco.2022.102567",
                "keywords": ["international trade", "economic growth", "developing countries", "trade liberalization"]
            },
            {
                "title": "Labor Market Dynamics and Unemployment Persistence",
                "abstract": "This study analyzes unemployment persistence across OECD countries, examining the role of labor market institutions, skills mismatches, and cyclical factors. We propose policy reforms to reduce structural unemployment while maintaining labor market flexibility.",
                "authors": "Anderson, L. and Williams, D.",
                "journal": "Labour Economics",
                "year": "2021",
                "doi": "10.1016/j.labeco.2021.101456",
                "keywords": ["unemployment", "labor markets", "OECD", "structural unemployment"]
            },
            {
                "title": "Climate Change and Agricultural Productivity: A Global Analysis",
                "abstract": "We assess the impact of climate change on agricultural productivity using satellite data and machine learning techniques. Our results indicate significant regional variations in climate effects, with important implications for food security and international trade patterns.",
                "authors": "Brown, A., Taylor, J., and Chen, L.",
                "journal": "Nature Economics",
                "year": "2023",
                "doi": "10.1038/s43016-023-00789-x",
                "keywords": ["climate change", "agriculture", "productivity", "food security"]
            },
            {
                "title": "Inflation Dynamics in the Post-Pandemic Era",
                "abstract": "This paper examines inflation dynamics following the COVID-19 pandemic, analyzing supply chain disruptions, fiscal stimulus effects, and changes in consumer behavior. We discuss implications for monetary policy frameworks and inflation targeting regimes.",
                "authors": "Thompson, R., Davis, M., and Kumar, A.",
                "journal": "Quarterly Journal of Economics",
                "year": "2023",
                "doi": "10.1093/qje/qjad012",
                "keywords": ["inflation", "pandemic", "supply chains", "monetary policy"]
            }
        ]
        
        retriever.add_economic_literature(sample_papers)
        
        # Add sample policy documents
        policy_docs = [
            {
                "id": "fed_policy_001",
                "text": "Federal Reserve Policy Statement: The Federal Open Market Committee decided to raise the federal funds rate by 0.25 percentage points to support the Committee's goal of maximum employment and price stability. The Committee expects that ongoing increases in the target range will be appropriate to bring inflation down to 2 percent over time.",
                "source": "Federal Reserve",
                "category": "Monetary Policy",
                "date": "2023-03-22",
                "tags": ["interest rates", "FOMC", "inflation targeting"]
            },
            {
                "id": "trade_policy_001", 
                "text": "Trade Policy Update: New bilateral trade agreement aims to reduce tariffs on manufactured goods by 15% over the next three years. The agreement is expected to boost bilateral trade by $50 billion annually and create jobs in export-oriented industries.",
                "source": "Department of Commerce",
                "category": "Trade Policy",
                "date": "2023-06-15",
                "tags": ["tariffs", "trade agreement", "manufacturing"]
            }
        ]
        
        retriever.add_documents(policy_docs)
    
    # Display collection statistics
    stats = retriever.get_collection_stats()
    logger.info(f"Knowledge base initialized successfully:")
    logger.info(f"  Total documents: {stats['total_documents']}")
    logger.info(f"  Categories: {stats['categories']}")
    logger.info(f"  Sources: {stats['sources']}")
    
    logger.info("RAG knowledge base setup completed")


if __name__ == "__main__":
    main() 