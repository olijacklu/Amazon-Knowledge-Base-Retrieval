import json
from pathlib import Path


def save_experiment_results(results_by_dataset, metrics_by_dataset, output_dir, experiment_name):
    """Save experiment results in consistent structure across all experiments"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exp_dir = output_path / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, results in results_by_dataset.items():
        qa_results_path = exp_dir / f"{dataset_name}_qa_results.json"
        with open(qa_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        chunks_path = exp_dir / f"{dataset_name}_chunks.json"
        chunks_data = [{'question_id': r['question_id'], 'retrieved_chunks': r['retrieved_chunks']} for r in results]
        with open(chunks_path, 'w') as f:
            json.dump(chunks_data, f, indent=2, default=str)
        
        aggregated_path = exp_dir / f"{dataset_name}_aggregated.json"
        with open(aggregated_path, 'w') as f:
            json.dump(metrics_by_dataset[dataset_name], f, indent=2, default=str)
    
    print(f"Results saved to: {exp_dir}")
