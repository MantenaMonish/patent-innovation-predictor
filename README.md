# Patent Innovation Predictor

A powerful agentic AI system for patent trend analysis and future technology prediction using Ollama, ChromaDB, and CrewAI.

## Overview

This system analyzes patent data to identify trends and predict future innovations in specific technology areas (with a focus on lithium battery technology). It uses a multi-agent approach with specialized roles for research direction, patent retrieval, data analysis, and innovation forecasting.

## System Architecture
```
┌───────────────────────────────────────────────────────────────┐
│                     User Interface Layer                      │
│                        (main.py)                              │
└───────────────────────────────────────────────────────────────┘
                │                │                │
                ▼                ▼                ▼
┌───────────────────────────────────────────────────────────────┐
│                 Agent Orchestration Layer                     │
│                      (agents.py)                              │
│  ┌──────────────────┐   ┌────────────────┐  ┌───────────────┐│
│  │ Research Director│   │Patent Retriever│  │Data Analyst   ││
│  └──────────────────┘   └────────────────┘  └───────────────┘│
│  ┌──────────────────┐                                         │
│  │Innovation        │                                         │
│  │Forecaster        │                                         │
│  └──────────────────┘                                         │
└───────────────────────────────────────────────────────────────┘
                │                │                │
                ▼                ▼                ▼
┌───────────────────────────────────────────────────────────────┐
│                Knowledge Processing Layer                     │
│                      (retrieval.py)                           │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐ │
│  │ Semantic      │    │ Hybrid        │    │ Iterative     │ │
│  │ Search        │    │ Search        │    │ Search        │ │
│  └───────────────┘    └───────────────┘    └───────────────┘ │
└───────────────────────────────────────────────────────────────┘
                │                │                │
                ▼                ▼                ▼
┌───────────────────────────────────────────────────────────────┐
│                     Data Storage Layer                        │
│  ┌───────────────────────────────────────────────────────────┐│
│  │                       ChromaDB                            ││
│  │              (Vector Database on port 8000)               ││
│  └───────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────┘
```

## Project Structure
```
researchCheck_agentic_pipeline/
├── docker-compose.yaml          # ChromaDB + Ollama containers
├── main.py                      # Main application entry point
├── agents.py                    # CrewAI agents and tasks
├── retrieval.py                 # Search functions (semantic, keyword, hybrid, iterative)
├── ingestion_tool.py            # Index patent data into ChromaDB
├── chromadb_client.py           # ChromaDB connection and collection management
├── embedding.py                 # Embedding using nomic-embed-text via Ollama
├── information_extracter.py     # Fetch patent data from SerpAPI
├── helper.py                    # SerpAPI helper functions
├── files/                       # Downloaded patent JSON files
├── .env                         # API keys (not committed to git)
└── requirements.txt             # Python dependencies
```

## Prerequisites

- Python 3.10+
- Docker Desktop running
- SerpAPI key for fetching patent data

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/researchCheck_agentic_pipeline.git
cd researchCheck_agentic_pipeline
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```
SERP_API_KEY=your_serpapi_key_here
```

### 5. Start Docker containers
```bash
docker compose up -d
```
This starts:
- **ChromaDB** on `http://localhost:8000` — vector database
- **Ollama** on `http://localhost:11434` — local LLM server

### 6. Pull required models
```bash
# Embedding model (required)
docker exec -it ollama ollama pull nomic-embed-text

# LLM for agents (choose one based on your RAM)
docker exec -it ollama ollama pull phi3        # 2.3GB — recommended
docker exec -it ollama ollama pull mistral     # 4.7GB — better quality
```

## Usage

### Step 1 — Fetch patent data
```bash
python information_extracter.py
# Enter search query: lithium battery
# Enter directory path: files
```

### Step 2 — Index data into ChromaDB
```bash
python ingestion_tool.py
```

### Step 3 — Verify data is loaded
```bash
python chromadb_client.py
# Should show: Available Collections: - patents
```

### Step 4 — Run the application
```bash
python main.py
```

### Menu Options
```
1. Run complete patent trend analysis and forecasting  ← CrewAI agents
2. Search for specific patents                         ← keyword/semantic/hybrid
3. Iterative patent exploration                        ← iterative search
4. View system status                                  ← health check
5. Exit
```

## How It Works

### Data Pipeline
```
SerpAPI (Google Patents)
        ↓
information_extracter.py  →  JSON files in /files
        ↓
ingestion_tool.py  →  embeddings via nomic-embed-text  →  ChromaDB
        ↓
retrieval.py  →  semantic/keyword/hybrid search
        ↓
agents.py  →  CrewAI multi-agent analysis
        ↓
main.py  →  results saved to patent_analysis_timestamp.txt
```

### Search Types
| Type | How it works |
|---|---|
| Keyword | Embedding search with document text filter |
| Semantic | Pure vector similarity using nomic-embed-text embeddings |
| Hybrid | Combines semantic + keyword, deduplicates results |
| Iterative | Refines query using top result titles across multiple steps |

### Agent Roles
| Agent | Role |
|---|---|
| Research Director | Defines research scope and plan |
| Patent Retriever | Fetches relevant patents from ChromaDB |
| Data Analyst | Identifies trends and patterns |
| Innovation Forecaster | Predicts future breakthroughs |

## Tech Stack

| Component | Technology |
|---|---|
| Vector Database | ChromaDB (Docker) |
| Embeddings | nomic-embed-text via Ollama |
| LLM | phi3 / mistral via Ollama |
| Agent Framework | CrewAI |
| Patent Data Source | SerpAPI (Google Patents) |
| Language | Python 3.10+ |

## Troubleshooting

**ChromaDB not connecting:**
```bash
docker compose ps        # check containers are running
docker compose up -d     # restart if needed
```

**Ollama model not found:**
```bash
docker exec -it ollama ollama list          # see available models
docker exec -it ollama ollama pull phi3     # pull if missing
```

**Agent timeout:**
- Use a lighter model: `phi3` instead of `mistral`
- Make sure no other heavy processes are running

**No patents found in search:**
```bash
python ingestion_tool.py    # re-index if collection is missing
```

## License
MIT