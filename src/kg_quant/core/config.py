"""
Configuration Management for KG-AgentQuant

Unified configuration loading and management with YAML support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy


class ConfigManager:
    """
    Configuration manager for KG-AgentQuant.

    Handles loading and merging of configuration files including:
    - Environment configuration
    - Data configuration
    - Experiment configuration
    - Model configuration
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._env_config: Optional[Dict] = None
        self._data_config: Optional[Dict] = None
        self._cache: Dict[str, Any] = {}

    def load_env(self) -> Dict:
        """Load environment configuration."""
        if self._env_config is None:
            self._env_config = self._load_yaml(self.config_dir / "env.yaml")
            self._env_config = self._resolve_env_vars(self._env_config)
        return deepcopy(self._env_config)

    def load_data(self) -> Dict:
        """Load data configuration."""
        if self._data_config is None:
            self._data_config = self._load_yaml(self.config_dir / "data.yaml")
        return deepcopy(self._data_config)

    def load_experiment(self, name: str) -> Dict:
        """Load experiment configuration."""
        cache_key = f"exp_{name}"
        if cache_key in self._cache:
            return deepcopy(self._cache[cache_key])

        exp_config = self._load_yaml(self.config_dir / "experiments" / f"{name}.yaml")
        exp_config = self._resolve_refs(exp_config)
        self._cache[cache_key] = exp_config
        return deepcopy(exp_config)

    def load_model(self, name: str) -> Dict:
        """Load model configuration."""
        cache_key = f"model_{name}"
        if cache_key in self._cache:
            return deepcopy(self._cache[cache_key])

        model_config = self._load_yaml(self.config_dir / "models" / f"{name}.yaml")
        self._cache[cache_key] = model_config
        return deepcopy(model_config)

    def get_data_path(self, dataset: str) -> str:
        """Get path for a specific dataset."""
        env = self.load_env()
        data = self.load_data()

        base_path = env.get("data_root", "./data")
        dataset_info = data["datasets"].get(dataset, {})

        qlib_data_dir = dataset_info.get("qlib_data_dir")
        if qlib_data_dir:
            if not qlib_data_dir.startswith("/"):
                path = f"{base_path}/{qlib_data_dir}"
            else:
                path = qlib_data_dir
        else:
            qlib_key = dataset_info.get("qlib_market", "cn_data")
            path = env.get("qlib", {}).get(qlib_key, "")
            if "${data_root}" in path:
                path = path.replace("${data_root}", base_path)

        return path

    def get_experiment_config_path(self, name: str) -> Path:
        """Get path to experiment configuration file."""
        return self.config_dir / "experiments" / f"{name}.yaml"

    def list_experiments(self) -> list:
        """List all available experiments."""
        exp_dir = self.config_dir / "experiments"
        if not exp_dir.exists():
            return []
        return [f.stem for f in exp_dir.glob("*.yaml")]

    def _load_yaml(self, path: Path) -> Dict:
        """Load YAML file."""
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _resolve_env_vars(self, config: Any) -> Any:
        """Resolve environment variables in configuration."""
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.environ.get(var_name, config)
        return config

    def _resolve_refs(self, config: Dict) -> Dict:
        """Resolve configuration references."""
        config = deepcopy(config)

        if "data" in config:
            data_cfg = config["data"]

            if "dataset" in data_cfg:
                dataset_name = data_cfg["dataset"]
                data_info = self.load_data()["datasets"].get(dataset_name, {})
                data_cfg.update(data_info)

            for period in ["train_period", "valid_period", "test_period"]:
                if period in data_cfg and isinstance(data_cfg[period], str):
                    period_name = data_cfg[period]
                    if period_name in data_info:
                        data_cfg[period] = data_info[period_name]

        if "model" in config:
            model_cfg = config["model"]
            if "config_file" in model_cfg:
                model_name = model_cfg["config_file"].replace(".yaml", "")
                model_info = self.load_model(model_name)
                model_cfg.update(model_info)
                del model_cfg["config_file"]

        return config


_config_manager_instance: Optional[ConfigManager] = None


def get_config_manager(config_dir: str = "configs") -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager(config_dir)
    return _config_manager_instance


__all__ = ['ConfigManager', 'get_config_manager']