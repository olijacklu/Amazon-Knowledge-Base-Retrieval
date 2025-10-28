import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def fuse_and_rank(union_chunks, alpha, k=5):
    """Fuse semantic and BM25 scores with alpha and return top-k chunks"""
    for chunk in union_chunks:
        fused_score = alpha * chunk['normalized_semantic_score'] + (1 - alpha) * chunk['normalized_bm25_score']
        chunk['fused_score'] = fused_score
    
    sorted_chunks = sorted(union_chunks, key=lambda x: x['fused_score'], reverse=True)
    return sorted_chunks[:k]


def format_for_metrics(top_chunks, ground_truth):
    """Format chunks for metrics computation"""
    retrieved_chunks = [{'text': c['content'], 'score': c['fused_score']} for c in top_chunks]
    chunk_scores = [c['fused_score'] for c in top_chunks]
    
    return {
        'retrieved_chunks': retrieved_chunks,
        'ground_truth': ground_truth,
        'chunk_scores': chunk_scores
    }


def compute_alpha_prediction_metrics(y_true, y_pred):
    """Compute metrics for alpha prediction"""
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_true': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_true': np.std(y_true),
        'std_pred': np.std(y_pred)
    }


def load_test_data(data_dir):
    """Load test set from merged retrievals"""
    
    all_data = []
    datasets = ['single_file', 'single_file_multihop', 'multi_file_multihop']
    
    for dataset in datasets:
        file_path = Path(data_dir) / f"{dataset}_merged.jsonl"
        if not file_path.exists():
            continue
        
        with open(file_path, 'r') as f:
            for line in f:
                item = eval(line.strip())
                item['dataset'] = dataset
                all_data.append(item)
    
    train_data, temp_data = train_test_split(
        all_data, test_size=0.3, random_state=42,
        stratify=[d['dataset'] for d in all_data]
    )
    _, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42,
        stratify=[d['dataset'] for d in temp_data]
    )
    
    return test_data


def compute_optimal_alpha(union_chunks):
    """Compute optimal alpha from union chunks"""
    if not union_chunks:
        return 0.5
    
    s = np.array([c['normalized_semantic_score'] for c in union_chunks])
    b = np.array([c['normalized_bm25_score'] for c in union_chunks])
    L = np.array([c['lss'] for c in union_chunks])
    
    diff = s - b
    numerator = np.sum((L - b) * diff)
    denominator = np.sum(diff ** 2)
    
    if denominator < 1e-10:
        return 0.5
    
    alpha = numerator / denominator
    return float(np.clip(alpha, 0.0, 1.0))
