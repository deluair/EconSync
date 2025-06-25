"""
Configuration management for EconSync system.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    
    vector_db_type: str = "chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
    collection_name: str = "econsync_knowledge"


@dataclass
class ReActConfig:
    """Configuration for ReAct framework."""
    
    max_iterations: int = 10
    reasoning_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    action_timeout: int = 30


@dataclass
class ModelConfig:
    """Configuration for base models."""
    
    base_model: str = "microsoft/DialoGPT-medium"
    tokenizer: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    load_in_8bit: bool = False


@dataclass
class DataConfig:
    """Configuration for data management."""
    
    data_dir: str = "data"
    cache_dir: str = ".cache"
    max_sequence_length: int = 2048
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EconSyncConfig:
    """Main configuration class for EconSync system."""
    
    # Sub-configurations
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    react: ReActConfig = field(default_factory=ReActConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # General settings
    project_name: str = "EconSync"
    version: str = "0.1.0"
    debug: bool = False
    seed: int = 42
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
        
        # Create directories if they don't exist
        Path(self.data.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "EconSyncConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EconSyncConfig":
        """Create configuration from dictionary."""
        # Extract sub-configurations
        lora_config = LoRAConfig(**config_dict.get("lora", {}))
        rag_config = RAGConfig(**config_dict.get("rag", {}))
        react_config = ReActConfig(**config_dict.get("react", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        # Remove sub-configurations from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ["lora", "rag", "react", "model", "data"]}
        
        return cls(
            lora=lora_config,
            rag=rag_config,
            react=react_config,
            model=model_config,
            data=data_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "lora": self.lora.__dict__,
            "rag": self.rag.__dict__,
            "react": self.react.__dict__,
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "project_name": self.project_name,
            "version": self.version,
            "debug": self.debug,
            "seed": self.seed,
        }
    
    def save_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        with open(config_path, 'w') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False) 