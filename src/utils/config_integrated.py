"""
Integrated Configuration Loader for CantioAI Complete System
Handles loading and merging of all stage configurations
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


class IntegratedConfigLoader:
    """Loads and manages integrated configuration for all CantioAI stages"""

    def __init__(self, base_config_path: str = "configs/integrated/cantioai.yaml"):
        self.base_config_path = Path(base_config_path)
        self.config_cache: Dict[str, Any] = {}
        self.loaded_references: set = set()

    def load_config(self) -> Dict[str, Any]:
        """Load the complete integrated configuration"""
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base configuration not found: {self.base_config_path}")

        # Load base configuration
        config = self._load_yaml_file(self.base_config_path)

        # Resolve all references
        config = self._resolve_references(config, self.base_config_path.parent)

        # Apply any overrides
        config = self._apply_overrides(config)

        # Validate configuration
        self._validate_config(config)

        return config

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single YAML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"[ERROR] Failed to load YAML file {file_path}: {e}")
            raise

    def _resolve_references(self, config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """Recursively resolve all configuration references"""
        if isinstance(config, dict):
            resolved_config = {}
            for key, value in config.items():
                if key == "reference_file" and isinstance(value, str):
                    # Handle direct file reference
                    ref_path = base_dir / value
                    if ref_path.exists():
                        ref_config = self._load_yaml_file(ref_path)
                        # Merge with any overrides in the same object
                        overrides = config.get("overrides", {})
                        if overrides:
                            ref_config = self._deep_merge(ref_config, overrides)
                        resolved_config.update(ref_config)
                    else:
                        logger.warning(f"[WARN] Reference file not found: {ref_path}")
                        resolved_config[key] = value  # Keep reference as-is
                elif key == "reference_configs" and isinstance(value, dict):
                    # Handle multiple config references
                    ref_section = {}
                    for ref_key, ref_path in value.items():
                        full_path = base_dir / ref_path
                        if full_path.exists():
                            if full_path.is_dir():
                                # Load all YAML files in directory
                                dir_config = {}
                                for yaml_file in full_path.glob("*.yaml"):
                                    file_config = self._load_yaml_file(yaml_file)
                                    dir_config[yaml_file.stem] = file_config
                                ref_section[ref_key] = dir_config
                            else:
                                # Single file
                                ref_section[ref_key] = self._load_yaml_file(full_path)
                        else:
                            logger.warning(f"[WARN] Reference config not found: {full_path}")
                            ref_section[ref_key] = {"error": f"Not found: {ref_path}"}
                    resolved_config.update(ref_section)
                elif key == "reference_section" and isinstance(value, str):
                    # Reference a section from the main config
                    main_config_path = base_dir.parent / "config.yaml"
                    if main_config_path.exists():
                        main_config = self._load_yaml_file(main_config_path)
                        if value in main_config:
                            resolved_config.update(main_config[value])
                        else:
                            logger.warning(f"[WARN] Section '{value}' not found in main config")
                            resolved_config[key] = value
                    else:
                        logger.warning(f"[WARN] Main config not found: {main_config_path}")
                        resolved_config[key] = value
                elif key == "existing_configs" and isinstance(value, dict):
                    # Handle existing config references (for backward compatibility)
                    existing_section = {}
                    for config_key, config_path in value.items():
                        full_path = base_dir.parent / config_path
                        if full_path.exists():
                            existing_section[config_key] = self._load_yaml_file(full_path)
                        else:
                            logger.warning(f"[WARN] Existing config not found: {full_path}")
                            existing_section[config_key] = {"error": f"Not found: {config_path}"}
                    resolved_config.update(existing_section)
                else:
                    # Recursively process nested structures
                    resolved_config[key] = self._resolve_references(value, base_dir)
            return resolved_config
        elif isinstance(config, list):
            return [self._resolve_references(item, base_dir) for item in config]
        else:
            return config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply system-wide overrides"""
        # Map quality presets to optimization levels
        if "inference" in config and "quality_preset" in config["inference"]:
            quality_preset = config["inference"]["quality_preset"]
            optimization_mapping = {
                "speed": "minimal",
                "balanced": "balanced",
                "quality": "aggressive",
                "max_quality": "extreme"
            }
            if "optimization" not in config:
                config["optimization"] = {}
            if "level" not in config["optimization"]:
                config["optimization"]["level"] = optimization_mapping.get(
                    quality_preset, "balanced"
                )

        # Map latency targets
        if "inference" in config and "optimization" in config:
            if "latency_targets" in config["optimization"] and "quality_preset" in config["inference"]:
                quality_preset = config["inference"]["quality_preset"]
                latency_targets = config["optimization"]["latency_targets"]
                target_latency = latency_targets.get(quality_preset, 30.0)
                config["inference"]["target_latency"] = target_latency

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Basic configuration validation"""
        required_sections = ["system", "model", "training", "inference", "webui", "deployment", "storage"]
        for section in required_sections:
            if section not in config:
                logger.warning(f"[WARN] Missing configuration section: {section}")

        # Validate system mode
        valid_modes = ["development", "production", "edge", "testing"]
        if config.get("system", {}).get("mode") not in valid_modes:
            logger.warning(f"[WARN] Invalid system mode: {config.get('system', {}).get('mode')}")

        # Validate device
        valid_devices = ["auto", "cpu"] + [f"cuda:{i}" for i in range(8)]  # Support up to 8 GPUs
        device = config.get("system", {}).get("device", "auto")
        if device not in valid_devices and not device.startswith("cuda:"):
            logger.warning(f"[WARN] Invalid device specification: {device}")


# Global config loader instance
_config_loader = None


def get_integrated_config() -> Dict[str, Any]:
    """Get the integrated configuration (singleton pattern)"""
    global _config_loader
    if _config_loader is None:
        _config_loader = IntegratedConfigLoader()
    return _config_loader.load_config()


def reload_config() -> Dict[str, Any]:
    """Reload the configuration (clears cache)"""
    global _config_loader
    _config_loader = IntegratedConfigLoader()
    return _config_loader.load_config()


# Convenience function for backward compatibility
def get_config() -> Dict[str, Any]:
    """Backward compatible config getter"""
    return get_integrated_config()


if __name__ == "__main__":
    # Test the configuration loader
    try:
        config = get_integrated_config()
        print("[PASS] Integrated configuration loaded successfully")
        print(f"  System mode: {config.get('system', {}).get('mode')}")
        print(f"  Device: {config.get('system', {}).get('device')}")
        print(f"  Model architecture: {config.get('model', {}).get('architecture')}")
        print(f"  Inference mode: {config.get('inference', {}).get('mode')}")
        print(f"  WebUI enabled: {config.get('webui', {}).get('enabled')}")
        print(f"  Target latency: {config.get('inference', {}).get('target_latency')} ms")
    except Exception as e:
        print(f"[FAIL] Failed to load integrated configuration: {e}")
        raise