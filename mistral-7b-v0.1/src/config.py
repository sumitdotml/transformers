"""
Configuration utilities for neural network models.
"""

import os
import json
from typing import Dict, Any

# Define a dictionary for hardcoded fallback default values
# These are used ONLY if configs/default.json is missing or unparsable.
_FALLBACK_DEFAULTS = {
    "model_type": "transformer",
    "hidden_dim": 512,
    "vocab_size": 30000,
    "output_dim": 10,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "num_attention_heads": 8,
    "ff_dim": 2048,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_function": "gelu",
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "warmup_steps": 10000,
    "epochs": 10,
    "optimizer": "adamw",
}


class ModelConfig:

    def __init__(self, **kwargs):
        """
        Initialize the configuration.

        Attributes are set dynamically based on provided kwargs.
        If no kwargs are provided, attempts to load from 'configs/default.json'.
        If that fails, uses hardcoded fallback default values.

        Args:
            **kwargs: Configuration values to set as attributes.
        """
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            # No kwargs, load from default.json or use hardcoded fallbacks
            self._load_system_defaults()

    def _load_system_defaults(self):
        """
        Loads default configuration values for the instance.
        Priority: configs/default.json > hardcoded _FALLBACK_DEFAULTS.
        """
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "default.json",
        )

        loaded_config_data = None
        if os.path.exists(default_config_path):
            try:
                with open(default_config_path, "r") as f:
                    loaded_config_data = json.load(f)
            except Exception as e:
                print(
                    f"Warning: Could not load or parse configs/default.json: {e}. Using fallback defaults."
                )
        else:
            print(
                f"Warning: configs/default.json not found. Using fallback default values."
            )

        # Use loaded data or fallback if loading failed
        final_config_data = (
            loaded_config_data if loaded_config_data is not None else _FALLBACK_DEFAULTS
        )

        for key, value in final_config_data.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns:
            Dictionary of configuration values (instance attributes).
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }

    def save_pretrained(self, config_dir: str) -> str:
        """
        Save the configuration to a JSON file named config.json.

        Args:
            config_dir: Directory to save the configuration

        Returns:
            Path to the saved configuration file
        """
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "config.json")

        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return config_path

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Create a configuration from a dictionary.

        Args:
            config_dict: Dictionary of configuration values

        Returns:
            ModelConfig instance
        """
        return cls(**config_dict)  # Pass dict items as kwargs to __init__

    @classmethod
    def from_json_file(cls, json_file: str) -> "ModelConfig":
        """
        Load a configuration from a JSON file.

        Args:
            json_file: Path to the JSON file

        Returns:
            ModelConfig instance
        """
        if not os.path.exists(json_file):
            print(
                f"Warning: Config file {json_file} not found. Initializing a default ModelConfig."
            )
            # This will attempt to load configs/default.json or use _FALLBACK_DEFAULTS
            return cls()

        try:
            with open(json_file, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            print(
                f"Warning: Could not load or parse {json_file}: {e}. Initializing a default ModelConfig."
            )
            return cls()

    @classmethod
    def from_pretrained(cls, model_path_or_config_file: str) -> "ModelConfig":
        """
        Load a configuration from a directory containing config.json,
        a direct path to a config.json, or fall back to system defaults.

        Args:
            model_path_or_config_file: Path to the directory or config file.

        Returns:
            ModelConfig instance
        """
        config_file_to_load = None

        # Check if it's a directory containing config.json
        if os.path.isdir(model_path_or_config_file):
            potential_config_path = os.path.join(
                model_path_or_config_file, "config.json"
            )
            if os.path.exists(potential_config_path):
                config_file_to_load = potential_config_path
        # Check if it's a direct path to a .json file
        elif os.path.isfile(
            model_path_or_config_file
        ) and model_path_or_config_file.endswith(".json"):
            config_file_to_load = model_path_or_config_file

        if config_file_to_load:
            return cls.from_json_file(config_file_to_load)
        else:
            print(
                f"Warning: No specific config.json found for '{model_path_or_config_file}'. "
                f"Attempting to load system default configuration (from configs/default.json or fallbacks)."
            )
            # This will attempt to load configs/default.json or use _FALLBACK_DEFAULTS
            return cls()

    def __repr__(self) -> str:
        """
        Get string representation of the configuration.

        Returns:
            String representation
        """
        return f"{self.__class__.__name__}({self.to_dict()})"


# Global instance of default configuration, loaded on module import.
# ModelConfig() when called with no args will attempt to load from 'configs/default.json' first.
DEFAULT_CONFIG = ModelConfig()
