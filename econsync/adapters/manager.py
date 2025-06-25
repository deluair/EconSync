"""
LoRA Adapter Manager for EconSync.
"""

from typing import Dict, List, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import logging

from ..core.config import EconSyncConfig
from ..utils.logger import setup_logger


class AdapterManager:
    """
    Manages multiple LoRA adapters for different economic domains.
    """
    
    def __init__(self, config: EconSyncConfig):
        """
        Initialize the adapter manager.
        
        Args:
            config: EconSync configuration
        """
        self.config = config
        self.logger = setup_logger("AdapterManager", debug=config.debug)
        
        # Available adapters
        self.available_adapters = {
            "trade_economics": "Trade Economics Analysis",
            "financial_markets": "Financial Markets Analysis", 
            "agricultural_economics": "Agricultural Economics",
            "policy_analysis": "Policy Impact Analysis",
            "macroeconomic_forecasting": "Macroeconomic Forecasting"
        }
        
        # Active adapters
        self.active_adapters: List[str] = []
        self.loaded_adapters: Dict[str, Any] = {}
        
        # Base model
        self.base_model = None
        self.tokenizer = None
        
        self._initialize_base_model()
    
    def _initialize_base_model(self):
        """Initialize the base model and tokenizer."""
        try:
            self.logger.info(f"Loading base model: {self.config.model.base_model}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.tokenizer,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model,
                torch_dtype=getattr(torch, self.config.model.torch_dtype),
                device_map=self.config.model.device,
                load_in_4bit=self.config.model.load_in_4bit,
                trust_remote_code=True
            )
            
            self.logger.info("Base model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise
    
    def create_adapter(self, adapter_name: str) -> None:
        """
        Create a new LoRA adapter for a specific domain.
        
        Args:
            adapter_name: Name of the adapter to create
        """
        if adapter_name not in self.available_adapters:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        self.logger.info(f"Creating adapter: {adapter_name}")
        
        try:
            # Configure LoRA for this adapter
            lora_config = LoraConfig(
                r=self.config.lora.rank,
                lora_alpha=self.config.lora.alpha,
                target_modules=self.config.lora.target_modules,
                lora_dropout=self.config.lora.dropout,
                bias=self.config.lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Create PEFT model
            peft_model = get_peft_model(self.base_model, lora_config)
            
            # Store the adapter
            self.loaded_adapters[adapter_name] = {
                "model": peft_model,
                "config": lora_config,
                "description": self.available_adapters[adapter_name]
            }
            
            self.logger.info(f"Adapter {adapter_name} created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create adapter {adapter_name}: {e}")
            raise
    
    def activate_adapters(self, adapter_names: List[str]) -> None:
        """
        Activate specific adapters for use.
        
        Args:
            adapter_names: List of adapter names to activate
        """
        self.logger.info(f"Activating adapters: {adapter_names}")
        
        for adapter_name in adapter_names:
            if adapter_name not in self.available_adapters:
                self.logger.warning(f"Unknown adapter: {adapter_name}")
                continue
            
            # Create adapter if not already loaded
            if adapter_name not in self.loaded_adapters:
                self.create_adapter(adapter_name)
            
            # Add to active list if not already active
            if adapter_name not in self.active_adapters:
                self.active_adapters.append(adapter_name)
        
        self.logger.info(f"Active adapters: {self.active_adapters}")
    
    def deactivate_adapter(self, adapter_name: str) -> None:
        """
        Deactivate a specific adapter.
        
        Args:
            adapter_name: Name of adapter to deactivate
        """
        if adapter_name in self.active_adapters:
            self.active_adapters.remove(adapter_name)
            self.logger.info(f"Deactivated adapter: {adapter_name}")
    
    def get_active_adapters(self) -> List[str]:
        """Get list of currently active adapters."""
        return self.active_adapters.copy()
    
    def get_available_adapters(self) -> Dict[str, str]:
        """Get list of all available adapters."""
        return self.available_adapters.copy()
    
    def generate_with_adapters(self, 
                              prompt: str, 
                              max_length: int = 512,
                              temperature: float = 0.7) -> str:
        """
        Generate text using active adapters.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Generation temperature
            
        Returns:
            Generated text
        """
        if not self.active_adapters:
            self.logger.warning("No adapters active, using base model")
            model = self.base_model
        else:
            # Use the first active adapter (can be enhanced for multi-adapter inference)
            adapter_name = self.active_adapters[0]
            model = self.loaded_adapters[adapter_name]["model"]
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """
        Get information about a specific adapter.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            Adapter information dictionary
        """
        if adapter_name not in self.loaded_adapters:
            return {"status": "not_loaded", "description": self.available_adapters.get(adapter_name, "Unknown")}
        
        adapter_info = self.loaded_adapters[adapter_name]
        return {
            "status": "loaded",
            "active": adapter_name in self.active_adapters,
            "description": adapter_info["description"],
            "config": adapter_info["config"].__dict__
        }
    
    def save_adapter(self, adapter_name: str, save_path: str) -> None:
        """
        Save a trained adapter to disk.
        
        Args:
            adapter_name: Name of adapter to save
            save_path: Path to save the adapter
        """
        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"Adapter {adapter_name} not loaded")
        
        self.logger.info(f"Saving adapter {adapter_name} to {save_path}")
        
        try:
            adapter = self.loaded_adapters[adapter_name]["model"]
            adapter.save_pretrained(save_path)
            self.logger.info(f"Adapter saved successfully to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save adapter: {e}")
            raise
    
    def load_adapter(self, adapter_name: str, load_path: str) -> None:
        """
        Load a trained adapter from disk.
        
        Args:
            adapter_name: Name to assign to the loaded adapter
            load_path: Path to load the adapter from
        """
        self.logger.info(f"Loading adapter {adapter_name} from {load_path}")
        
        try:
            # Load the adapter
            model = PeftModel.from_pretrained(self.base_model, load_path)
            
            self.loaded_adapters[adapter_name] = {
                "model": model,
                "config": model.peft_config,
                "description": f"Loaded from {load_path}"
            }
            
            self.logger.info(f"Adapter loaded successfully from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load adapter: {e}")
            raise 