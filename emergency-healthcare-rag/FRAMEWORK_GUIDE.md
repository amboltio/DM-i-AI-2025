# Emergency Healthcare RAG - Framework Guide

## Overview

All frameworks (Ollama, llama.cpp, vLLM, Transformers) are indeed based on transformers, but they serve different purposes and have different performance characteristics.

## Framework Comparison

### 1. **vLLM** - Best for Production Performance

**Best for**: High-throughput inference, production deployments

**Advantages**:

- Fastest inference (2-4x faster than standard Transformers)
- Efficient memory management with PagedAttention
- Excellent for batch processing
- Built-in quantization support

**Memory Usage**: Optimized for large models, can handle 7B-70B models efficiently

**Setup**:

```bash
pip install vllm
```

**Usage Example**:

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="microsoft/DialoGPT-medium", gpu_memory_utilization=0.8)

# Inference
sampling_params = SamplingParams(temperature=0.1, max_tokens=100)
outputs = llm.generate(["Your medical statement here"], sampling_params)
```

### 2. **Transformers (Hugging Face)** - Most Flexible

**Best for**: Research, fine-tuning, custom architectures

**Advantages**:

- Most flexible and customizable
- Excellent for fine-tuning
- Wide model support
- Easy to modify and extend

**Memory Usage**: Higher memory usage, but very flexible

**Setup**:

```bash
pip install transformers torch
```

**Usage Example** (already implemented in our `model.py`):

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
```

### 3. **llama.cpp** - Best for Resource-Constrained Environments

**Best for**: CPU inference, low-resource environments

**Advantages**:

- Excellent memory efficiency with GGUF quantization
- Can run on CPU efficiently
- Very low VRAM usage
- Good for edge deployment

**Memory Usage**: Very low with proper quantization (can run 7B models on 8GB RAM)

**Setup**:

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# For GPU support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

**Usage Example**:

```python
from llama_cpp import Llama

# Load quantized model
llm = Llama(
    model_path="./models/llama-2-7b-chat.gguf",
    n_ctx=2048,
    n_gpu_layers=35
)

# Inference
output = llm("Your medical statement here", max_tokens=100)
```

### 4. **Ollama** - Easiest to Use

**Best for**: Quick prototyping, easy deployment

**Advantages**:

- Easiest to set up and use
- Built-in model management
- Good performance with optimization
- Simple API

**Memory Usage**: Moderate, well-optimized

**Setup**:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2:7b
```

**Usage Example**:

```python
import ollama

# Inference
response = ollama.chat(model='llama2:7b', messages=[
    {
        'role': 'user',
        'content': 'Your medical statement here'
    }
])
```

## Using Pretrained Weights Instead of API Calls

### Method 1: Download and Use Local Models

```python
from transformers import AutoTokenizer, AutoModel

# Download and cache model locally
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
model = AutoModel.from_pretrained(model_name, cache_dir="./models")

# Save locally for offline use
tokenizer.save_pretrained("./local_model")
model.save_pretrained("./local_model")

# Load from local path
tokenizer = AutoTokenizer.from_pretrained("./local_model")
model = AutoModel.from_pretrained("./local_model")
```

### Method 2: Use Quantized Models for Lower Memory

```python
# For 4-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModel.from_pretrained(
    "microsoft/DialoGPT-medium",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Training on Your Dataset

### Step 1: Prepare Your Data

The training script (`train_model.py`) already handles this:

- Loads statements from `data/train/statements/`
- Loads labels from `data/train/answers/`
- Creates proper dataset splits

### Step 2: Choose a Base Model

For medical tasks, consider these models:

- `microsoft/DialoGPT-medium` (current choice)
- `microsoft/BioGPT` (medical domain)
- `GanjinZero/COGMEN` (medical entity recognition)
- `dmis-lab/biobert-base-cased-v1.2` (biomedical)

### Step 3: Fine-tune the Model

```bash
python train_model.py
```

### Step 4: Use Your Fine-tuned Model

```python
# In model.py, update the model path
model_path = "fine_tuned_medical_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
```

## Recommended Approach for Your Constraints

Given your constraints (5 seconds max, 24GB VRAM, offline):

### Option 1: vLLM with Quantized Model (Recommended)

```python
from vllm import LLM, SamplingParams

# Use a quantized model for lower memory usage
llm = LLM(
    model="microsoft/DialoGPT-medium",
    quantization="awq",  # or "gptq"
    gpu_memory_utilization=0.8,
    max_model_len=2048
)
```

### Option 2: llama.cpp with GGUF Model

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/medical-model.gguf",
    n_ctx=2048,
    n_gpu_layers=35,
    n_batch=512
)
```

### Option 3: Transformers with 4-bit Quantization

```python
from transformers import BitsAndBytesConfig, AutoModel

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModel.from_pretrained(
    "fine_tuned_medical_model",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Performance Optimization Tips

1. **Model Quantization**: Use 4-bit or 8-bit quantization to reduce memory usage
2. **Batch Processing**: Process multiple statements together when possible
3. **Context Length**: Limit context length to what's necessary
4. **Caching**: Cache embeddings and model outputs where possible
5. **GPU Memory**: Monitor GPU memory usage and adjust batch sizes accordingly

## Memory Usage Guidelines

- **24GB VRAM**: Can handle 7B-13B models with quantization
- **16GB VRAM**: Can handle 7B models comfortably
- **8GB VRAM**: Use llama.cpp with GGUF quantization
- **CPU Only**: Use llama.cpp or quantized models

Choose the framework based on your specific hardware constraints and performance requirements!
