"""
Core Module

Provides core framework components.
"""

from .evaluator import Evaluator
from .config import ConfigManager, get_config_manager

__all__ = ['Evaluator', 'ConfigManager', 'get_config_manager']