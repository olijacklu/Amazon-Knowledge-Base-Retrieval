import argparse
import json
import numpy as np
from pathlib import Path

from ..evaluation_utils import load_test_data, compute_optimal_alpha, fuse_and_rank, format_for_metrics, compute_alpha_prediction_metrics
from ...evaluation.metrics import compute_metrics
from ...evaluation.aggregator import aggregate_metrics
from ...shared.save_experiment_results import save_experiment_results


FIXED_ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def main():
    parser = argparse.ArgumentParser(description='Evaluate fixed alpha baselines')
    parser.add_argument('--input-dir', default='results/hybrid_retrieval', help='Input directory with merged.jsonl files')
    parser.add_argument('--output-dir', default='results/hybrid_retrieval/fixed_alpha_eval', help='Output directory')
    parser.add_argument('--md-dir', default='data/converted_md', help='Markdown directory')
    args = parser.parse_args()
    
    print("=== FIXED ALPHA BASELINES EVALUATION ===")
    
    test_data = load_test_data(args.input_dir)
    print(f"Loaded {len(test_data)} test samples")
    
    all_results = {}
    
    for fixed_alpha in FIXED_ALPHAS:
        print(f"\n=== Testing α = {fixed_alpha} ===")
        
        y_true = []
        y_pred = []
        results = []
        
        for i, item in enumerate(test_data):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(test_data)}")
            
            optimal_alpha = compute_optimal_alpha(item['union_chunks'])
            
            y_true.append(optimal_alpha)
            y_pred.append(fixed_alpha)
            
            top_chunks = fuse_and_rank(item['union_chunks'], fixed_alpha, k=5)
            formatted = format_for_metrics(top_chunks, item['ground_truth'])
            
            result = {
                'question_id': i,
                'question': item['question'],
                'predicted_alpha': fixed_alpha,
                'optimal_alpha': optimal_alpha,
                'dataset': item['dataset'],
                **formatted
            }
            results.append(result)
        
        print("Computing retrieval metrics...")
        document_cache = {}
        for result in results:
            result['metrics'] = compute_metrics(
                result['retrieved_chunks'],
                result['ground_truth'],
                args.md_dir,
                result.get('chunk_scores'),
                document_cache
            )
        
        results_by_dataset = {}
        for result in results:
            dataset = result['dataset']
            if dataset not in results_by_dataset:
                results_by_dataset[dataset] = []
            results_by_dataset[dataset].append(result)
        
        metrics_by_dataset = {}
        for dataset, dataset_results in results_by_dataset.items():
            metrics_by_dataset[dataset] = aggregate_metrics(dataset_results)
        
        aggregated = aggregate_metrics(results)
        
        save_experiment_results(results_by_dataset, metrics_by_dataset, args.output_dir, f'alpha_{fixed_alpha}')
        
        print("Computing alpha prediction metrics...")
        alpha_metrics = compute_alpha_prediction_metrics(np.array(y_true), np.array(y_pred))
        
        print(f"\nAlpha Prediction:")
        print(f"  MSE: {alpha_metrics['mse']:.6f}")
        print(f"  MAE: {alpha_metrics['mae']:.6f}")
        
        print(f"\nRetrieval Metrics:")
        print(f"  T1EM: {aggregated['t1em']:.4f}")
        print(f"  T5EM: {aggregated['t5em']:.4f}")
        print(f"  MRR: {aggregated['mrr']:.4f}")
        print(f"  BC: {aggregated['bc']:.4f}")
        
        alpha_dir = Path(args.output_dir) / f'alpha_{fixed_alpha}'
        with open(alpha_dir / "alpha_metrics.json", 'w') as f:
            json.dump(alpha_metrics, f, indent=2)
        
        all_results[f"alpha_{fixed_alpha}"] = {
            'alpha_metrics': alpha_metrics,
            'retrieval_metrics': aggregated
        }
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    print(f"{'Alpha':<10} {'T1EM':<10} {'MRR':<10} {'BC':<10} {'MAE':<10}")
    print("-" * 50)
    for fixed_alpha in FIXED_ALPHAS:
        key = f"alpha_{fixed_alpha}"
        metrics = all_results[key]
        print(f"{fixed_alpha:<10.1f} {metrics['retrieval_metrics']['t1em']:<10.4f} "
              f"{metrics['retrieval_metrics']['mrr']:<10.4f} {metrics['retrieval_metrics']['bc']:<10.4f} "
              f"{metrics['alpha_metrics']['mae']:<10.4f}")
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
