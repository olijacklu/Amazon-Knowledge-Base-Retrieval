import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the full KB pipeline")
    parser.add_argument('--skip-data-processing', action='store_true', help='Skip PDF conversion and cleaning')
    parser.add_argument('--skip-qa-generation', action='store_true', help='Skip QA generation')
    parser.add_argument('--experiment', choices=['chunking', 'graph', 'hybrid', 'reranking', 'all'], 
                       default='all', help='Which experiments to run')
    parser.add_argument('--chunking-method', choices=['fixed_window', 'semantic', 'hierarchical', 'agentic_1', 'agentic_2'],
                       help='Specific chunking method to run')
    args = parser.parse_args()
    
    # Data Processing
    if not args.skip_data_processing:
        run_command(
            "python -m src.data_processing.convert_to_markdown",
            "Step 1: Converting PDFs to Markdown"
        )
        run_command(
            "python -m src.data_processing.clean_markdowns",
            "Step 2: Cleaning Markdown files"
        )
    
    # QA Generation
    if not args.skip_qa_generation:
        run_command(
            "python -m src.qa_generation.single_file_generator",
            "Step 3: Generating single-file QAs"
        )
        
        if Path('data/paper_tags.json').exists():
            run_command(
                "python -m src.qa_generation.multi_file_generator --tags-file data/paper_tags.json",
                "Step 4: Generating multi-hop QAs"
            )
    
    # Experiments
    if args.experiment in ['chunking', 'all']:
        methods = [args.chunking_method] if args.chunking_method else ['fixed_window', 'semantic', 'hierarchical']
        for method in methods:
            run_command(
                f"python -m src.experiments.chunking.{method}.setup",
                f"Running {method} chunking setup"
            )
            run_command(
                f"python -m src.experiments.chunking.{method}.evaluate",
                f"Evaluating {method} chunking"
            )
    
    if args.experiment in ['graph', 'all']:
        for method in ['graph_hierarchical', 'graph_agentic']:
            run_command(
                f"python -m src.experiments.graph_retrieval.{method}.setup",
                f"Running {method} setup"
            )
            run_command(
                f"python -m src.experiments.graph_retrieval.{method}.evaluate",
                f"Evaluating {method}"
            )
    
    if args.experiment in ['hybrid', 'all']:
        run_command(
            "python -m src.experiments.hybrid_retrieval.precompute_retrievals",
            "Precomputing hybrid retrievals"
        )
        for method in ['ridge', 'xgboost', 'mlp']:
            run_command(
                f"python -m src.experiments.hybrid_retrieval.{method}.train",
                f"Training {method} fusion model"
            )
            run_command(
                f"python -m src.experiments.hybrid_retrieval.{method}.evaluate",
                f"Evaluating {method} fusion"
            )
    
    if args.experiment in ['reranking', 'all']:
        run_command(
            "python -m src.experiments.conditional_reranking.compute_fused_scores",
            "Computing fused scores for reranking"
        )
        run_command(
            "python -m src.experiments.conditional_reranking.llm_rerank",
            "Running LLM reranking"
        )
        run_command(
            "python -m src.experiments.conditional_reranking.calibrate_threshold",
            "Calibrating reranking threshold"
        )
        run_command(
            "python -m src.experiments.conditional_reranking.evaluate",
            "Evaluating conditional reranking"
        )
    
    print(f"\n{'='*80}")
    print("Pipeline completed successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
