# 🌍 EconSync: Smart Agent for Applied Economics Research

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deluair/EconSync/blob/main/notebooks/EconSync_Colab_Demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Overview

EconSync is a sophisticated AI agent system that integrates **Low-Rank Adaptation (LoRA)**, **Retrieval-Augmented Generation (RAG)**, and **Reasoning and Acting (ReAct)** frameworks to revolutionize applied economics research. The system combines parameter-efficient fine-tuning, knowledge grounding, and autonomous reasoning capabilities to create an intelligent research assistant for economic analysis.

### 🚀 Key Features

- **🔧 Multiple LoRA Adapters**: Specialized adapters for trade economics, financial markets, agricultural economics, policy analysis, and macroeconomic forecasting
- **📚 RAG Knowledge Infrastructure**: Multi-modal retrieval system for academic literature, government statistics, and financial data
- **🤖 ReAct Reasoning Framework**: Autonomous reasoning and acting capabilities for complex economic research
- **📊 Synthetic Data Generation**: Comprehensive economic datasets for training and testing
- **⚡ Real-time Analysis**: Live market analysis and policy impact assessment
- **🔬 Advanced Analytics**: Automated econometric modeling and literature review

## 💰 Cost Analysis & Deployment Options

### 🎯 Development & Testing Costs

| Platform | Monthly Cost | GPU/CPU | RAM | Best For |
|----------|-------------|---------|-----|----------|
| **Google Colab Free** | $0 | T4 (limited) | 12.7GB | Development/Learning |
| **Google Colab Pro** | $9.99 | T4/P100 | 12.7GB | Small Research Projects |
| **Google Colab Pro+** | $49.99 | V100/TPU | 25.5GB | Medium Research Projects |

### 🏢 Production Deployment Costs

| Platform | Monthly Cost | Specs | Best For |
|----------|-------------|-------|----------|
| **AWS EC2 (g4dn.xlarge)** | $526 | T4 GPU, 16GB RAM | GPU Production |
| **AWS EC2 (p3.2xlarge)** | $3,060 | V100 GPU, 61GB RAM | Heavy ML Training |
| **Azure ML (Standard_NC6)** | $900 | K80 GPU, 56GB RAM | Enterprise GPU |
| **GCP Vertex AI** | $146 | T4 GPU, 15GB RAM | ML Pipeline |

### 📊 Additional Operational Costs

| Service | Monthly Cost Range | Usage |
|---------|-------------------|-------|
| OpenAI API (GPT-4) | $20-200 | LLM inference calls |
| Anthropic Claude | $15-150 | Alternative LLM calls |
| FRED API | Free | Economic data access |
| Financial Data APIs | $100-1,000 | Real-time market data |
| Vector Database Storage | $10-100 | Knowledge base storage |
| Monitoring & Logging | $25-100 | System operations |

## 🚀 Quick Start

### 📋 Option 1: Google Colab (Recommended for Testing)

1. **Click the Colab badge above** to open the demo notebook
2. **Run all cells** to see EconSync in action
3. **Experiment** with your own economic questions

### 💻 Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/deluair/EconSync.git
cd EconSync

# Create virtual environment
python -m venv econsync_env
source econsync_env/bin/activate  # On Windows: econsync_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Generate synthetic datasets
python scripts/generate_datasets.py

# Initialize vector database
python scripts/setup_rag.py
```

### 🔧 Option 3: Docker Deployment

```bash
# Build the Docker image
docker build -t econsync:latest .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data econsync:latest

# Access the web interface
open http://localhost:8000
```

## 📖 Usage Examples

### 🐍 Python API

```python
from econsync import EconSyncAgent

# Initialize the agent
agent = EconSyncAgent()

# Load economic data
agent.load_data("macroeconomic_indicators")

# Run comprehensive analysis
result = agent.analyze("What are the inflation trends for 2024?")
print(result)

# Generate forecasts
forecast = agent.forecast(
    variable="inflation",
    horizon=30,  # 30 days ahead
    method="arima"
)

# Policy impact assessment
policy_result = agent.policy_impact(
    "Increase interest rates by 0.5%",
    affected_sectors=["housing", "automotive"]
)
```

### 🖥️ Command Line Interface

```bash
# Initialize EconSync
econsync init

# Run analysis
econsync analyze "Examine trade relationships between US and China"

# Generate forecasts
econsync forecast --variable inflation --horizon 90 --method arima

# Policy analysis
econsync policy "Carbon tax implementation" --sectors energy,manufacturing

# Check system status
econsync status
```

### 🌐 Web Interface

```bash
# Start the web server
econsync serve --port 8000

# Access the dashboard
open http://localhost:8000
```

## 🏗️ Project Structure

```
EconSync/
├── 📦 econsync/                    # Core package
│   ├── 🧠 adapters/               # LoRA adapters management
│   ├── 📚 rag/                    # RAG infrastructure  
│   ├── 🤖 react/                  # ReAct framework
│   ├── 📊 data/                   # Data management and generators
│   ├── 📈 models/                 # Economic models
│   ├── 🔬 analytics/              # Analysis tools
│   ├── ⚙️ core/                   # Configuration and main agent
│   └── 🛠️ utils/                  # Utilities and logging
├── 📁 data/                       # Generated datasets
├── ⚙️ configs/                    # Configuration files
├── 🧪 tests/                      # Test suite
├── 📜 scripts/                    # Setup and utility scripts
├── 💡 examples/                   # Usage examples
├── 📓 notebooks/                  # Jupyter tutorials
└── 📖 docs/                       # Documentation
```

## 🔧 Configuration

EconSync uses YAML configuration files for flexible setup:

```yaml
# configs/default.yaml
model:
  base_model: "microsoft/DialoGPT-large"
  device: "auto"
  max_length: 1024

lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  adapters:
    - trade_economics
    - financial_markets
    - macroeconomic_forecasting

rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_db_path: "./chroma_db"
  top_k: 10

react:
  max_iterations: 5
  confidence_threshold: 0.8
```

## 📊 Synthetic Datasets

EconSync includes comprehensive synthetic dataset generation:

### 📈 Macroeconomic Indicators (10 years, daily)
- GDP growth, inflation (CPI, PCE, core)
- Unemployment rates, labor force participation
- Interest rates, monetary policy measures
- Productivity and business cycle indicators

### 🌍 International Trade Data
- Bilateral trade flows (goods/services)
- Tariff rates and trade policies
- Export/import price indices
- Trade balance and competitiveness measures

### 🏢 Firm-Level Microdata (50,000 firms)
- Revenue, employment, capital stock
- R&D expenditure, trade participation
- Sector classification (NAICS)
- Geographic distribution

### 🏛️ Policy & Institutional Data
- 10,000 regulatory changes with impact assessments
- Trade agreements and dispute resolution
- Central bank communications with sentiment analysis

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_agent.py
pytest tests/test_data_generators.py

# Run with coverage
pytest --cov=econsync tests/
```

## 📊 Performance Benchmarks

Based on our Colab testing:

- **Data Processing**: 1,000+ samples/second
- **Analysis Speed**: 0.1-2.0 seconds per query
- **Memory Usage**: 50-200MB for typical datasets
- **GPU Utilization**: 30-80% on T4 (Google Colab)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **💾 Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **📤 Push** to the branch (`git push origin feature/amazing-feature`)
5. **🔃 Open** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformers and PEFT libraries
- **LangChain** for RAG infrastructure components
- **ChromaDB** for vector database capabilities
- **Federal Reserve (FRED)** for economic data standards
- **OpenAI & Anthropic** for foundation model inspirations

## 📞 Support & Community

- **📧 Issues**: [GitHub Issues](https://github.com/deluair/EconSync/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/deluair/EconSync/discussions)
- **📖 Documentation**: [Full Documentation](https://deluair.github.io/EconSync)
- **🎓 Tutorials**: [Jupyter Notebooks](notebooks/)

## 🚀 Future Roadmap

### 🔄 Short Term (Q1 2024)
- [ ] Integration with real-time data sources (FRED, Yahoo Finance)
- [ ] Web-based dashboard interface
- [ ] Additional LoRA adapters for specialized domains
- [ ] Performance optimizations and quantization

### 🎯 Medium Term (Q2-Q3 2024)
- [ ] Multi-modal analysis (charts, images, documents)
- [ ] Advanced econometric modeling capabilities
- [ ] Collaborative research features
- [ ] Enterprise security and compliance features

### 🌟 Long Term (Q4 2024+)
- [ ] Mobile applications for real-time analysis
- [ ] Integration with major economic databases
- [ ] Advanced AI agents for autonomous research
- [ ] Global economic simulation capabilities

---

**EconSync: Democratizing AI-powered economic research** 🌍📊🤖

*Made with ❤️ by the EconSync Team* 