# Agentic Retrieval for Knowledge Bases: From Chunking to Reranking

Experimental framework for evaluating enterprise knowledge base retrieval methods with a focus on accuracy-latency-cost trade-offs for agentic systems.

**Full thesis:** [Agentic Retrieval for Knowledge Bases.pdf](Agentic%20Retrieval%20for%20Knowledge%20Bases.pdf)

## Background

Large organizations generate extensive technical documentation (SOPs, research reports, internal procedures) that is domain-specific and unavailable during LLM pre-training. This creates challenges for retrieval-augmented generation (RAG) systems that power enterprise agents.

This project addresses four critical implementation decisions for enterprise RAG:

1. **Document Segmentation** - How to chunk long, structured documents (fixed-window vs. semantic vs. hierarchical vs. agentic)
2. **Graph Structure** - Whether knowledge graphs improve multi-hop reasoning over flat retrieval
3. **Hybrid Retrieval** - How to balance sparse (BM25) and dense (vector) signals, with query-aware fusion
4. **Reranking** - When to apply expensive LLM reranking vs. using fusion scores directly

## Key Contributions

- **Agentic Chunking**: LLM-guided segmentation with contextual summaries embedded for retrieval
- **Fair Graph-Augmented Retrieval Evaluation**: Apples-to-apples comparison of graph methods vs. best chunking baselines on multi-hop questions
- **Query-Aware Hybrid Fusion**: Learned per-query weights (Ridge Regression/XGBoost/MLP) for BM25+vector combination
- **Conditional Reranking**: Calibrated threshold policy to trigger LLM reranking only when needed

Unlike Wikipedia-based benchmarks, this evaluates on long-form, multimodal enterprise documents with complex structure.

## Prerequisites

- Python 3.8>=
- AWS account with Bedrock access
- AWS credentials with permissions for:
  - Amazon Bedrock (Claude models, Titan embeddings)
  - Amazon S3 (for data storage)
  - AWS Neptune (for graph experiments)
- ~10GB disk space for datasets

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and dependencies
pip install -e .

# Configure AWS credentials
cp .env.example .env
# Edit .env with your AWS credentials

# Download data from Google Drive
python download_data.py
```

This downloads PDFs, markdown files, paper_tags.json, and QA datasets to `data/`.

## Quick Start

```bash
# Run full pipeline
python run_pipeline.py

# Skip data processing (if using downloaded data)
python run_pipeline.py --skip-data-processing --skip-qa-generation

# Run specific experiments
python run_pipeline.py --experiment chunking --chunking-method fixed_window
python run_pipeline.py --experiment hybrid
python run_pipeline.py --experiment reranking
```

## Pipeline

### 1. Data Processing

```bash
# Convert PDFs to markdown (if not using downloaded data)
python -m src.data_processing.convert_to_markdown

# Clean markdown files
python -m src.data_processing.clean_markdowns
```

### 2. QA Generation

```bash
# Generate single-file QAs
python -m src.qa_generation.single_file_generator

# Generate multi-hop QAs (requires paper_tags.json)
python -m src.qa_generation.multi_file_generator --tags-file data/paper_tags.json
```

### 3. Run Experiments

**Chunking experiments:**
```bash
# Fixed window chunking
python -m src.experiments.chunking.fixed_window.setup
python -m src.experiments.chunking.fixed_window.evaluate

# Semantic chunking
python -m src.experiments.chunking.semantic.setup
python -m src.experiments.chunking.semantic.evaluate

# Hierarchical chunking
python -m src.experiments.chunking.hierarchical.setup
python -m src.experiments.chunking.hierarchical.evaluate

# Agentic chunking (variant 1: enhanced)
python -m src.experiments.chunking.agentic_1.setup
python -m src.experiments.chunking.agentic_1.evaluate

# Agentic chunking (variant 2: adaptive)
python -m src.experiments.chunking.agentic_2.setup
python -m src.experiments.chunking.agentic_2.evaluate
```

**Graph retrieval:**
```bash
# Graph with hierarchical chunking
python -m src.experiments.graph_retrieval.graph_hierarchical.setup
python -m src.experiments.graph_retrieval.graph_hierarchical.evaluate

# Graph with agentic chunking
python -m src.experiments.graph_retrieval.graph_agentic.setup
python -m src.experiments.graph_retrieval.graph_agentic.evaluate
```

**Hybrid retrieval:**
```bash
# Precompute BM25 and vector retrievals
python -m src.experiments.hybrid_retrieval.precompute_retrievals

# Fixed alpha baseline
python -m src.experiments.hybrid_retrieval.fixed_alpha.evaluate

# Ridge regression fusion
python -m src.experiments.hybrid_retrieval.ridge.train
python -m src.experiments.hybrid_retrieval.ridge.evaluate

# XGBoost fusion
python -m src.experiments.hybrid_retrieval.xgboost.train
python -m src.experiments.hybrid_retrieval.xgboost.evaluate

# MLP fusion
python -m src.experiments.hybrid_retrieval.mlp.train
python -m src.experiments.hybrid_retrieval.mlp.evaluate
```

**Conditional reranking:**
```bash
# Compute fused scores from hybrid retrieval
python -m src.experiments.conditional_reranking.compute_fused_scores

# Apply LLM reranking
python -m src.experiments.conditional_reranking.llm_rerank

# Calibrate threshold for conditional policy
python -m src.experiments.conditional_reranking.calibrate_threshold

# Evaluate conditional reranking
python -m src.experiments.conditional_reranking.evaluate
```

## Configuration

AWS credentials are configured via `.env` file:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

## Project Structure

```
amazon_kb/
├── src/                                    # Source code
│   ├── __init__.py                        # Auto-loads .env for AWS credentials
│   ├── data_processing/                   # PDF→markdown conversion
│   │   ├── convert_to_markdown.py         # PDF to markdown using Docling
│   │   ├── clean_markdowns.py             # Remove images, clean formatting
│   │   └── entity_extraction.py           # Extract shared entities across papers
│   ├── qa_generation/                     # QA dataset generation
│   │   ├── llm_client.py                  # AWS Bedrock Claude API client
│   │   ├── single_file_generator.py       # Generate single-file QAs
│   │   ├── multi_file_generator.py        # Generate multi-hop QAs
│   │   └── text_processing.py             # Text cleaning and chunking
│   ├── evaluation/                        # Metrics and aggregation
│   │   ├── metrics.py                     # 11 retrieval metrics (t1em, t5em, etc.)
│   │   └── aggregator.py                  # Aggregate metrics across datasets
│   └── experiments/
│       ├── chunking/                      # Document segmentation experiments
│       │   ├── fixed_window/              # Fixed-size sliding window
│       │   ├── semantic/                  # Layout-aware semantic chunking
│       │   ├── hierarchical/              # Parent-child hierarchical chunks
│       │   ├── agentic_1/                 # LLM-guided enhanced chunking
│       │   ├── agentic_2/                 # LLM-guided adaptive chunking
│       │   ├── base_experiment.py         # Shared KB creation logic
│       │   └── agentic_utils.py           # Agentic chunking utilities
│       ├── graph_retrieval/               # Knowledge graph experiments
│       │   ├── graph_hierarchical/        # Graph + hierarchical chunks
│       │   ├── graph_agentic/             # Graph + agentic chunks
│       │   └── graph_utils.py             # Neptune KB creation utilities
│       ├── hybrid_retrieval/              # BM25 + vector fusion
│       │   ├── precompute_retrievals.py   # Precompute BM25 and vector results
│       │   ├── fixed_alpha/               # Fixed-weight baseline
│       │   ├── ridge/                     # Ridge regression fusion
│       │   ├── xgboost/                   # XGBoost fusion
│       │   ├── mlp/                       # MLP fusion
│       │   └── evaluation_utils.py        # Shared evaluation functions
│       ├── conditional_reranking/         # LLM reranking with calibration
│       │   ├── compute_fused_scores.py    # Add fusion scores to candidates
│       │   ├── llm_rerank.py              # Claude-based reranking
│       │   ├── calibrate_threshold.py     # Wilson score calibration
│       │   └── evaluate.py                # Test multiple thresholds
│       └── shared/                        # Shared experiment utilities
│           ├── evaluate_kb.py             # Standard KB evaluation
│           ├── save_experiment_results.py # Save results in standard format
│           ├── kb_utils.py                # Load evaluation data
│           └── lambda_utils.py            # AWS Lambda utilities
├── data/                                  # Data directory (gitignored)
│   ├── pdf/                               # Source PDF files
│   ├── converted_md/                      # Converted markdown files
│   ├── qas/                               # Single-file QA datasets
│   ├── multihop_qas/                      # Multi-hop QA datasets
│   └── paper_tags.json                    # Paper topic mappings
├── results/                               # Experiment outputs (gitignored)
│   ├── chunking/                          # Chunking experiment results
│   ├── graph_retrieval/                   # Graph experiment results
│   ├── hybrid_retrieval/                  # Hybrid fusion results
│   └── conditional_reranking/             # Reranking results
├── .env.example                           # AWS credentials template
├── .gitignore                             # Git ignore rules
├── setup.py                               # Package installation
├── requirements.txt                       # Python dependencies
├── download_data.py                       # Download datasets from Google Drive
├── run_pipeline.py                        # Main pipeline orchestration
└── README.md                              # This file
```

### Directory Organization

**Source Code (`src/`):**
- All Python modules organized by function
- Each experiment has `setup.py` (create KB) and `evaluate.py` (run evaluation)
- Shared utilities in `experiments/shared/` to avoid code duplication

**Data (`data/`):**
- Downloaded via `download_data.py`
- Not tracked in git (see `.gitignore`)
- Organized by processing stage: pdf → converted_md → qas

**Results (`results/`):**
- Generated by experiments
- Not tracked in git
- Each experiment creates subdirectories with:
  - `{dataset}_qa_results.json` - Full results with metrics
  - `{dataset}_chunks.json` - Retrieved chunks per question
  - `{dataset}_aggregated.json` - Aggregated metrics
  - `kb_ids.json` - Knowledge base identifiers (for chunking/graph experiments)

## Acknowledgments

I would like to personally thank Buğra Çetingök, Samuele Compagnoni, and Carlos Moyano for all of their support and guidance throughout my internship at Amazon.

Furtheremore, special thanks to Professor Vincent Lepetit for his supervision and guidance.
