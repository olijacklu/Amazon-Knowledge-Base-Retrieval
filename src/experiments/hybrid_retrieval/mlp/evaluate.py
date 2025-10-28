import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from ..evaluation_utils import load_test_data, compute_optimal_alpha, fuse_and_rank, format_for_metrics, compute_alpha_prediction_metrics
from .train import FusionMLP, get_query_embedding
from .config import INPUT_DIM
from ...evaluation.metrics import compute_metrics
from ...evaluation.aggregator import aggregate_metrics
from ...shared.save_experiment_results import save_experiment_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MLP model')
    parser.add_argument('--input-dir', default='results/hybrid_retrieval', help='Input directory with merged.jsonl files')
    parser.add_argument('--model-dir', default='results/hybrid_retrieval/models', help='Model directory')
    parser.add_argument('--output-dir', default='results/hybrid_retrieval/mlp_eval', help='Output directory')
    parser.add_argument('--md-dir', default='data/converted_md', help='Markdown directory')
    args = parser.parse_args()
    
    print("=== MLP EVALUATION ===")
    
    model_path = Path(args.model_dir) / "mlp_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    best_params = checkpoint['best_params']
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionMLP(
        input_dim=INPUT_DIM,
        hidden_dims=best_params['hidden_dims'],
        dropout=best_params['dropout'],
        use_batch_norm=best_params.get('use_batch_norm', False)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Best params: {best_params}")
    
    test_data = load_test_data(args.input_dir)
    print(f"Loaded {len(test_data)} test samples")
    
    y_true = []
    y_pred = []
    results = []
    
    print("\nPredicting alpha and computing metrics...")
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_data)}")
        
        embedding = get_query_embedding(item['question'])
        X = torch.FloatTensor([embedding]).to(device)
        
        with torch.no_grad():
            predicted_alpha = model(X).item()
        
        predicted_alpha = np.clip(predicted_alpha, 0.0, 1.0)
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
    
    save_experiment_results(results_by_dataset, metrics_by_dataset, args.output_dir, 'mlp')
    
    output_path = Path(args.output_dir) / 'mlp'
    
    if train_losses and val_losses:
        print("\n=== PLOTTING LOSS CURVES ===")
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('MLP Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / "loss_curves.png", dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to {output_path / 'loss_curves.png'}")
    
    with open(output_path / "alpha_metrics.json", 'w') as f:
        json.dump(alpha_metrics, f, indent=2)
    
    with open(output_path / "best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    with open(output_path / "training_history.json", 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
