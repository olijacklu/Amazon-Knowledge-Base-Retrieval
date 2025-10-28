import argparse
import json
import numpy as np
import joblib
from pathlib import Path

from ..evaluation_utils import load_test_data, compute_optimal_alpha, fuse_and_rank, format_for_metrics, compute_alpha_prediction_metrics
from .config import FEATURE_NAMES, WH_WORDS
from ...evaluation.metrics import compute_metrics
from ...evaluation.aggregator import aggregate_metrics
from ...shared.save_experiment_results import save_experiment_results


def extract_simple_features(question):
    """Extract 4 simple features from question"""
    words = question.lower().split()
    
    return {
        'word_count': len(words),
        'has_numbers': float(any(char.isdigit() for char in question)),
        'has_wh': float(any(word in words for word in WH_WORDS)),
        'has_acronyms': float(any(word.isupper() and len(word) > 1 for word in question.split()))
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ridge model')
    parser.add_argument('--input-dir', default='results/hybrid_retrieval', help='Input directory with merged.jsonl files')
    parser.add_argument('--model-dir', default='results/hybrid_retrieval/models', help='Model directory')
    parser.add_argument('--output-dir', default='results/hybrid_retrieval/ridge_eval', help='Output directory')
    parser.add_argument('--md-dir', default='data/converted_md', help='Markdown directory')
    args = parser.parse_args()
    
    print("=== RIDGE EVALUATION ===")
    
    model_path = Path(args.model_dir) / "ridge_model.joblib"
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    print(f"Loaded model from {model_path}")
    
    test_data = load_test_data(args.input_dir)
    print(f"Loaded {len(test_data)} test samples")
    
    y_true = []
    y_pred = []
    results = []
    
    print("\nPredicting alpha and computing metrics...")
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_data)}")
        
        features = extract_simple_features(item['question'])
        X = np.array([[features[name] for name in feature_names]])
        X_scaled = scaler.transform(X)
        
        predicted_alpha = np.clip(model.predict(X_scaled)[0], 0.0, 1.0)
        optimal_alpha = compute_optimal_alpha(item['union_chunks'])
        
        y_true.append(optimal_alpha)
        y_pred.append(predicted_alpha)
        
        top_chunks = fuse_and_rank(item['union_chunks'], predicted_alpha, k=5)
        formatted = format_for_metrics(top_chunks, item['ground_truth'])
        
        result = {
            'question_id': i,
            'question': item['question'],
            'predicted_alpha': predicted_alpha,
            'optimal_alpha': optimal_alpha,
            'dataset': item['dataset'],
            **formatted
        }
        results.append(result)
    
    print("\nComputing retrieval metrics...")
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
    
    print("\nComputing alpha prediction metrics...")
    alpha_metrics = compute_alpha_prediction_metrics(np.array(y_true), np.array(y_pred))
    
    print("\n=== ALPHA PREDICTION ===")
    print(f"MSE: {alpha_metrics['mse']:.6f}")
    print(f"MAE: {alpha_metrics['mae']:.6f}")
    print(f"RMSE: {alpha_metrics['rmse']:.6f}")
    print(f"R²: {alpha_metrics['r2']:.4f}")
    print(f"Mean α (true): {alpha_metrics['mean_true']:.4f} ± {alpha_metrics['std_true']:.4f}")
    print(f"Mean α (pred): {alpha_metrics['mean_pred']:.4f} ± {alpha_metrics['std_pred']:.4f}")
    
    print("\n=== RETRIEVAL METRICS ===")
    print(f"T1EM: {aggregated['t1em']:.4f}")
    print(f"T5EM: {aggregated['t5em']:.4f}")
    print(f"MRR: {aggregated['mrr']:.4f}")
    print(f"BC: {aggregated['bc']:.4f}")
    
    print("\n=== FEATURE COEFFICIENTS ===")
    coef_abs = np.abs(model.coef_)
    for name, coef, abs_coef in sorted(zip(feature_names, model.coef_, coef_abs), key=lambda x: x[2], reverse=True):
        print(f"{name}: {coef:.4f} (|{abs_coef:.4f}|)")
    
    save_experiment_results(results_by_dataset, metrics_by_dataset, args.output_dir, 'ridge')
    
    output_path = Path(args.output_dir) / 'ridge'
    
    with open(output_path / "alpha_metrics.json", 'w') as f:
        json.dump(alpha_metrics, f, indent=2)
    
    with open(output_path / "coefficients.json", 'w') as f:
        json.dump({name: float(coef) for name, coef in zip(feature_names, model.coef_)}, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
