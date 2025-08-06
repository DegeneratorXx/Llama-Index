# LlamaIndex RAG System

A Retrieval-Augmented Generation (RAG) system built with LlamaIndex for document-based question answering using OpenAI and HuggingFace models.

## Overview

This project demonstrates how to build a complete RAG pipeline that can:
- Load and process documents
- Create vector embeddings for semantic search
- Answer questions based on document content
- Support multiple LLM and embedding providers

## Project Structure

```
LlamaIndex/
├── HuggingFace-OpenAI-LlamaIndex/     # Main notebook directory
│   ├── HF-locally.ipynb              # Initial experiments with local HF models
│   └── OpenAI-llm.ipynb              # Working OpenAI implementation
├── data/                              # Document storage
│   └── paul_graham/
│       └── paul_graham_essay.txt      # Sample document
├── tutorial/                          # Tutorial files
│   ├── attention.pdf                 # Sample PDF document
│   └── Basics.ipynb                  # Basic tutorial implementation
├── .env                              # Environment variables (API keys)
├── .gitignore                        # Git ignore rules
├── README.md                         # Project documentation
└── requirement.txt                   # Project dependencies
```

### Directory Descriptions

#### HuggingFace-OpenAI-LlamaIndex/
Contains the main implementation notebooks:
- **HF-locally.ipynb**: Experiments with local HuggingFace models including TinyLlama
- **OpenAI-llm.ipynb**: implementation using OpenAI GPT-3.5-turbo

#### data/
- **paul_graham_essay.txt**: Sample document used for RAG demonstrations and testing

#### tutorial/
Learning materials and examples:
- **attention.pdf**: Additional PDF document for testing
- **Basics.ipynb**: Basic tutorial and learning implementation

#### Configuration Files
- **.env**: Contains API keys for OpenAI and HuggingFace
- **requirement.txt**: Python package dependencies
- **.gitignore**: Specifies files to ignore in version control


## Features

- **Document Processing**: Automatic text chunking and preprocessing
- **Vector Embeddings**: Semantic search using OpenAI or HuggingFace embeddings
- **Multiple LLM Support**: OpenAI GPT models and HuggingFace models
- **Query Engine**: Natural language question answering
- **Configurable Settings**: Adjustable chunk sizes, temperature, and model parameters

## System Architecture

### Document Processing Flow
```
Raw Document (Paul Graham Essay)
    ↓
SimpleDirectoryReader("./data/paul_graham/")
    ↓
Document Objects Collection
    ↓
Text Splitter (chunk_size=512 tokens)
    ↓
Document Chunks Array
    ↓
OpenAI Embedding Model (text-embedding-ada-002)
    ↓
High-dimensional Vector Representations
    ↓
VectorStoreIndex.from_documents()
    ↓
In-Memory Vector Database
```

### Query Processing Flow
```
User Query: "what does author say about html?"
    ↓
Query Text → OpenAI Embedding Model → Query Vector
    ↓
Cosine Similarity Computation
    ↓
Top-K Most Relevant Document Chunks
    ↓
Context Assembly
    ↓
Formatted Prompt:
"Context: [chunk1][chunk2][chunk3]
 Question: [user_query]
 Answer:"
    ↓
Context + Query → GPT-3.5-turbo → Generated Response
    ↓
Final Answer → User Interface
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LlamaIndex
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
touch .env

# Add your API keys
echo "OPENAI_API_KEY=sk-your-openai-key-here" >> .env
echo "HF_TOKEN=hf_your-huggingface-token-here" >> .env
```

## Usage

### Basic Setup

```python
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Configure models
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.embed_model = OpenAIEmbedding()
Settings.chunk_size = 512

# Load documents
docs = SimpleDirectoryReader("./data/paul_graham/").load_data()

# Create index and query engine
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

# Query the documents
response = query_engine.query("What does the author say about HTML?")
print(response)
```

### Alternative: HuggingFace Local Models

```python
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

# Local HuggingFace models (no API key required)
llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    context_window=2048,
    max_new_tokens=256,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = llm
Settings.embed_model = embed_model
```

## Configuration

### Model Options

#### Language Models
- **OpenAI**: GPT-3.5-turbo, GPT-4 (recommended for production)
- **HuggingFace Local**: TinyLlama, Microsoft DialoGPT, DistilGPT-2
- **HuggingFace API**: Mistral-7B, Llama-2, Zephyr-7B

#### Embedding Models
- **OpenAI**: text-embedding-ada-002 (recommended)
- **HuggingFace**: sentence-transformers/all-MiniLM-L6-v2 (free, local)

### Settings

```python
# Chunk configuration
Settings.chunk_size = 512          # Tokens per chunk
Settings.chunk_overlap = 50        # Overlap between chunks

# LLM parameters
temperature = 0.1                  # Response randomness (0.0-1.0)
max_new_tokens = 256              # Maximum response length

# Retrieval settings
similarity_top_k = 2              # Number of chunks to retrieve
```

## API Keys

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Create new API key
3. Add to `.env` file as `OPENAI_API_KEY=sk-...`

### HuggingFace Token
1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create new token
3. Add to `.env` file as `HF_TOKEN=hf_...`

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure `OPENAI_API_KEY` is set in `.env` file
   - Check API key validity and billing status

2. **HuggingFace Model Not Found**
   - Verify model name exists on HuggingFace Hub
   - Check HuggingFace token permissions

3. **Memory Issues with Local Models**
   - Use smaller models (TinyLlama, DistilGPT-2)
   - Enable `torch_dtype=torch.float16` for memory efficiency

### Performance Optimization

```python
# Memory efficient settings
Settings.chunk_size = 256          # Smaller chunks
model_kwargs = {"torch_dtype": torch.float16}  # Half precision

# Faster embeddings
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # Use GPU if available
)
```

## Examples

See the notebook files for complete examples:
- `OpenAI-llm.ipynb`: Production-ready OpenAI implementation
- `Basic.ipynb`: Local HuggingFace model experiments
- `Hf-API.ipynb`: HuggingFace API integration attempts

## Dependencies

Core dependencies:
- `llama-index`: Main framework
- `openai`: OpenAI API client
- `transformers`: HuggingFace model loading
- `torch`: PyTorch for local models
- `sentence-transformers`: Embedding models
- `python-dotenv`: Environment variable management

See `requirements.txt` for complete dependency list.
