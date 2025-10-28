import argparse
import json
import numpy as np
import joblib
import shap
from pathlib import Path

from ..evaluation_utils import load_test_data, compute_optimal_alpha, fuse_and_rank, format_for_metrics, compute_alpha_prediction_metrics
from .config import WH_WORDS
from ...evaluation.metrics import compute_metrics
from ...evaluation.aggregator import aggregate_metrics
from ...shared.save_experiment_results import save_experiment_results


def extract_xgboost_features(question, union_chunks):
    """Extract 25 features"""
    words = question.lower().split()
    
    query_features = {
        'word_count': len(words),
        'has_numbers': float(any(char.isdigit() for char in question)),
        'has_wh': float(any(word in words for word in WH_WORDS)),
        'has_acronyms': float(any(word.isupper() and len(word) > 1 for word in question.split()))
    }
    
    semantic_chunks = [(c['semantic_score'], c['bm25_score']) for c in union_chunks if c['semantic_score'] > 0]
    bm25_chunks = [(c['semantic_score'], c['bm25_score']) for c in union_chunks if c['bm25_score'] > 0]
    
    semantic_chunks.sort(key=lambda x: x[0], reverse=True)
    bm25_chunks.sort(key=lambda x: x[1], reverse=True)
    
    semantic_scores = [c[0] for c in semantic_chunks[:10]] + [0.0] * (10 - len(semantic_chunks[:10]))
    bm25_scores = [c[1] for c in bm25_chunks[:10]] + [0.0] * (10 - len(bm25_chunks[:10]))
    
    semantic_ids = {c['chunk_id'] for c in union_chunks if c['semantic_score'] > 0}
    bm25_ids = {c['chunk_id'] for c in union_chunks if c['bm25_score'] > 0}
    overlap_count = len(semantic_ids & bm25_ids)
    
    features = query_features.copy()
    for i, score in enumerate(semantic_scores, 1):
        features[f'semantic_score_{i}'] = score
    for i, score in enumerate(bm25_scores, 1):
        features[f'bm25_score_{i}'] = score
    features['overlap_count'] = overlap_count
    
    return features


def main():
    parser = argparse.ArgumentParser(description='Evaluate XGBoost model')
    parser.add_argument('--input-dir', default='results/hybrid_retrieval', help='Input directory with merged.jsonl files')
    parser.add_argument('--model-dir', default='results/hybrid_retrieval/models', help='Model directory')
    parser.add_argument('--output-dir', default='results/hybrid_retrieval/xgboost_eval', help='Output directory')
    parser.add_argument('--md-dir', default='data/converted_md', help='Markdown directory')
    args = parser.parse_args()
    
    print("=== XGBOOST EVALUATION ===")
    
    model_path = Path(args.model_dir) / "xgboost_model.joblib"
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    print(f"Loaded model from {model_path}")
    
    test_data = load_test_data(args.input_dir)
    print(f"Loaded {len(test_data)} test samples")
    
    y_true = []
    y_pred = []
    X_test = []
    results = []
    
    print("\nPredicting alpha and computing metrics...")
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_data)}")
        
        features = extract_xgboost_features(item['question'], item['union_chunks'])
        X = np.array([[features[name] for name in feature_names]])
        X_test.append(X[0])
        
        predicted_alpha = np.clip(model.predict(X)[0], 0.0, 1.0)
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
    
    X_test = np.array(X_test)
    
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
    
    print("\n=== FEATURE IMPORTANCE ===")
    importance = model.feature_importances_
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for name, imp in feature_importance[:10]:
        print(f"{name}: {imp:.4f}")
    
    print("\n=== COMPUTING SHAP VALUES ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    
    print("Top 10 features by mean |SHAP|:")
    for name, shap_val in shap_importance[:10]:
        print(f"{name}: {shap_val:.4f}")
    
    save_experiment_results(results_by_dataset, metrics_by_dataset, args.output_dir, 'xgboost')
    
    output_path = Path(args.output_dir) / 'xgboost'
    
    with open(output_path / "alpha_metrics.json", 'w') as f:
        json.dump(alpha_metrics, f, indent=2)
    
    with open(output_path / "feature_importance.json", 'w') as f:
        json.dump({name: float(imp) for name, imp in feature_importance}, f, indent=2)
    
    with open(output_path / "shap_importance.json", 'w') as f:
        json.dump({name: float(shap_val) for name, shap_val in shap_importance}, f, indent=2)
    
    np.save(output_path / "shap_values.npy", shap_values)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
