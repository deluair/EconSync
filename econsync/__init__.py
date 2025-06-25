"""
EconSync: Smart Agent for Applied Economics Research

A sophisticated AI agent system that integrates LoRA, RAG, and ReAct frameworks
for revolutionary applied economics research.
"""

__version__ = "0.1.0"
__author__ = "EconSync Team"
__email__ = "team@econsync.ai"

from .core.agent import EconSyncAgent
from .core.config import EconSyncConfig
from .data.generators import DataGenerator
from .models.economic import EconomicModel
from .analytics.analyzer import EconomicAnalyzer

__all__ = [
    "EconSyncAgent",
    "EconSyncConfig", 
    "DataGenerator",
    "EconomicModel",
    "EconomicAnalyzer",
] 