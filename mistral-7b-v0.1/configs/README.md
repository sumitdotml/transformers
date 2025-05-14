# Neural Network Project Configuration Guide

This guide explains how to use and create configuration files for your neural network projects based on this template.

## Overview

Configuration in this template is handled by JSON files (e.g., `default.json`, `my_experiment.json`) located in the `configs/` directory. These files define the parameters for your model architecture, training process, data handling, and more.

The core configuration loading mechanism is provided by the `ModelConfig` class in `src/config.py`. It dynamically loads attributes from these JSON files.

## `default.json`

The `configs/default.json` file provides a baseline set of parameters. When you initialize a `ModelConfig` object without specifying a particular file (e.g., when `DEFAULT_CONFIG` in `src/config.py` is created), it will attempt to load its values from `configs/default.json`.

If `configs/default.json` is missing or unparsable, `ModelConfig` will fall back to a set of hardcoded default values defined within `src/config.py` itself (and print a warning).

## Creating Custom Configurations

For different experiments or model variations, you should create new JSON configuration files in the `configs/` directory (e.g., `configs/my_transformer_config.json`).

You can start by copying `default.json` and then modifying the parameters as needed for your specific experiment.

Example: `configs/my_custom_model.json`
```json
{
  "model_type": "my_custom_transformer",
  "hidden_dim": 512,
  "vocab_size": 30000,
  "num_encoder_layers": 8,
  "num_attention_heads": 8,
  "dropout": 0.1,
  "batch_size": 16,
  "learning_rate": 3e-5
  // ... other parameters as needed
}
```

## Core Configuration Parameters

The template now provides transformer-specific default parameters. Here are the key parameters found in `default.json`:

### Model Architecture Parameters
* `model_type` (str): Identifier for the model architecture (e.g., "transformer", "base_model").
* `hidden_dim` (int): Dimension of the hidden layers throughout the model (often called d_model in transformers).
* `vocab_size` (int): Size of the vocabulary for token embeddings.
* `output_dim` (int): Dimension of the model's output (e.g., number of classes for classification).
* `num_encoder_layers` (int): Number of transformer encoder layers to use.
* `num_decoder_layers` (int): Number of transformer decoder layers to use.
* `num_attention_heads` (int): Number of attention heads in multi-head attention.
* `ff_dim` (int): Hidden dimension of the feed-forward network in transformer layers.
* `max_position_embeddings` (int): Maximum sequence length supported by the positional embeddings.

### Regularization and Normalization Parameters
* `dropout` (float): Dropout probability for general model components.
* `attention_dropout` (float): Dropout probability specific to attention mechanisms.
* `layer_norm_eps` (float): Epsilon value for layer normalization stability.

### Training Parameters
* `batch_size` (int): Number of samples per batch.
* `learning_rate` (float): Learning rate for the optimizer.
* `weight_decay` (float): Weight decay (L2 regularization) coefficient.
* `adam_beta1` (float): Beta1 parameter for Adam/AdamW optimizer.
* `adam_beta2` (float): Beta2 parameter for Adam/AdamW optimizer.
* `adam_epsilon` (float): Epsilon parameter for Adam/AdamW optimizer.
* `warmup_steps` (int): Number of warmup steps for learning rate scheduling.
* `epochs` (int): Number of training epochs.
* `optimizer` (str): Type of optimizer (e.g., "adamw", "adam").

## Transformer-Specific Configuration

For transformer models, you may see additional parameters beyond those in `default.json`. For example, `configs/transformer_example.json` includes:

```json
{
  "model_type": "transformer",
  "hidden_dim": 768,
  "vocab_size": 50265,
  "output_dim": 2,
  "num_encoder_layers": 12,
  "num_decoder_layers": 0,
  "num_attention_heads": 12,
  "ff_dim": 3072,
  "dropout": 0.1,
  "attention_dropout": 0.1,
  "activation_function": "gelu",
  "layer_norm_eps": 1e-5,
  "max_position_embeddings": 1024,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "weight_decay": 0.01,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8,
  "warmup_steps": 500,
  "epochs": 3,
  "optimizer": "adamw",
  "gradient_accumulation_steps": 2,
  "max_grad_norm": 1.0,
  "use_encoder_only": true,
  "classifier_dropout": 0.1
}
```

Some additional transformer-specific parameters:

* `activation_function` (str): Activation function in transformer feed-forward networks (common values: "gelu", "relu").
* `use_encoder_only` (bool): Whether to use only the encoder part of the transformer (like BERT).
* `classifier_dropout` (float): Dropout specific to classification heads.
* `gradient_accumulation_steps` (int): Number of steps to accumulate gradients before updating weights.
* `max_grad_norm` (float): Maximum gradient norm for gradient clipping.

## Adding Model-Specific Parameters

Since the `ModelConfig` class loads attributes dynamically, you can add any parameter your specific model architecture requires directly into its JSON configuration file.

For example, if your custom transformer model needs `num_attention_heads` and `num_encoder_layers`:

```json
// In configs/my_transformer_config.json
{
  "model_type": "my_transformer",
  "hidden_dim": 768,
  "output_dim": 10,
  "num_attention_heads": 12,
  "num_encoder_layers": 6,
  // ... other parameters ...
}
```

Your model's `__init__` method can then access these as `config.num_attention_heads`.

## Obtaining Configuration from Hugging Face Transformers

Many modern neural network architectures are available in the Hugging Face Transformers library. You can often get a good starting point for your configuration parameters by inspecting the configuration of a pretrained model from their hub.

Here's how you can do it:

1.  **Install Transformers**: If you haven't already:
    ```bash
    pip install transformers
    ```

2.  **Load a Pretrained Model's Configuration in Python**:
    ```python
    from transformers import AutoConfig

    # Replace 'bert-base-uncased' with the Hugging Face model ID you're interested in
    model_id = "bert-base-uncased"
    
    try:
        # Load the configuration
        hf_config = AutoConfig.from_pretrained(model_id)
        
        # Print the configuration as a dictionary
        print(f"Configuration for {model_id}:")
        config_dict = hf_config.to_dict()
        
        # Pretty print (optional, but helpful for readability)
        import json
        print(json.dumps(config_dict, indent=2))
        
        # Now you can see all the parameters and their values.
        # You can pick the relevant ones and adapt them for your project's config.json.
        # For example:
        # my_project_config = {
        #   "model_type": "bert_custom", // Your identifier
        #   "hidden_dim": hf_config.hidden_size,
        #   "num_encoder_layers": hf_config.num_hidden_layers,
        #   "num_attention_heads": hf_config.num_attention_heads,
        #   "vocab_size": hf_config.vocab_size, // If relevant for your model
        #   // ... and so on
        # }
        
    except Exception as e:
        print(f"Could not load configuration for {model_id}: {e}")
    ```

3.  **Adapt and Use**: 
    *   Run this Python script.
    *   Copy the printed dictionary or relevant key-value pairs.
    *   Paste and adapt them into your project's JSON configuration file (e.g., `configs/my_bert_like_model.json`).
    *   You might need to rename some keys or select only the subset relevant to your specific implementation.

This process can save you time and provide sensible default values for complex architectures.

## Accessing Configuration in Code

In your Python scripts (e.g., training script, model definition):

```python
from src.config import ModelConfig

# Load a specific configuration
config = ModelConfig.from_pretrained("configs/my_experiment.json")

# Or get the default configuration (loads from configs/default.json by default)
# config = ModelConfig() 
# or
# from src.config import DEFAULT_CONFIG
# config = DEFAULT_CONFIG

# Access parameters
learning_rate = config.learning_rate
model_type = config.model_type
num_encoder_layers = config.num_encoder_layers

if hasattr(config, 'my_custom_param'):
    custom_param = config.my_custom_param
    print(f"Custom parameter found: {custom_param}")

# Your model initialization
# model = YourModelClass(config)
```

Remember to adjust paths and parameters according to your project's needs. 