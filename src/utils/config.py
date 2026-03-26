"""
Configuration loading utilities for CantioAI.
"""

import yaml
from typing import Dict, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    def _merge(base_dict, override_dict):
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                _merge(base_dict[key], value)
            else:
                base_dict[key] = value

    _merge(result, override)
    return result


def get_nested_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get nested value from configuration using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., "training.learning_rate")
        default: Default value if key path not found

    Returns:
        Value at key path or default
    """
    keys = key_path.split('.')
    current = config

    try:
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except Exception:
        return default