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


## Configuration

### Model Options

#### Language Models
- **OpenAI**: GPT-3.5-turbo, GPT-4 (recommended for production)
- **HuggingFace Local**: TinyLlama, Microsoft DialoGPT, DistilGPT-2
- **HuggingFace API**: Mistral-7B, Llama-2, Zephyr-7B

#### Embedding Models
- **OpenAI**: text-embedding-ada-002 (recommended)
- **HuggingFace**: sentence-transformers/all-MiniLM-L6-v2 (free, local)


## API Keys

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Create new API key
3. Add to `.env` file as `OPENAI_API_KEY=sk-...`

### HuggingFace Token
1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create new token
3. Add to `.env` file as `HF_TOKEN=hf_...`




## Dependencies

See `requirements.txt` for complete dependency list.
