import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

from .config import PARAM_GRID, WH_WORDS


def extract_xgboost_features(question, union_chunks):
    """Extract 25 features: 4 query + 10 semantic scores + 10 BM25 scores + 1 overlap"""
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


def compute_optimal_alpha(union_chunks):
    """Find optimal alpha by minimizing MSE between fused scores and LSS"""
    if not union_chunks:
        return 0.5, float('inf')
    
    s = np.array([c['normalized_semantic_score'] for c in union_chunks])
    b = np.array([c['normalized_bm25_score'] for c in union_chunks])
    L = np.array([c['lss'] for c in union_chunks])
    
    diff = s - b
    numerator = np.sum((L - b) * diff)
    denominator = np.sum(diff ** 2)
    
    if denominator < 1e-10:
        alpha = 0.5
    else:
        alpha = numerator / denominator
        alpha = np.clip(alpha, 0.0, 1.0)
    
    fused = alpha * s + (1 - alpha) * b
    mse = np.mean((fused - L) ** 2)
    
    return float(alpha), float(mse)


def load_data(data_dir):
    """Load all datasets and prepare training data"""
    print("Loading data...")
    
    all_data = []
    datasets = ['single_file', 'single_file_multihop', 'multi_file_multihop']
    
    for dataset in datasets:
        file_path = Path(data_dir) / f"{dataset}_merged.jsonl"
        if not file_path.exists():
            print(f"  Skipping {dataset} - file not found")
            continue
        
        print(f"  Loading {dataset}...")
        
        with open(file_path, 'r') as f:
            for line in f:
                item = eval(line.strip())
                
                features = extract_xgboost_features(item['question'], item['union_chunks'])
                optimal_alpha, mse = compute_optimal_alpha(item['union_chunks'])
                
                all_data.append({
                    'features': features,
                    'optimal_alpha': optimal_alpha,
                    'mse': mse,
                    'dataset': dataset
                })
    
    print(f"Loaded {len(all_data)} samples")
    return all_data


def prepare_features(data, feature_names):
    """Convert data to feature matrix and target vector"""
    X = np.array([[d['features'][name] for name in feature_names] for d in data])
    y = np.array([d['optimal_alpha'] for d in data])
    return X, y


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--input-dir', default='results/hybrid_retrieval', help='Input directory with merged.jsonl files')
    parser.add_argument('--output-dir', default='results/hybrid_retrieval/models', help='Output directory')
    args = parser.parse_args()
    
    print("=== XGBOOST TRAINING ===")
    
    data = load_data(args.input_dir)
    
    feature_names = ['word_count', 'has_numbers', 'has_wh', 'has_acronyms']
    feature_names += [f'semantic_score_{i}' for i in range(1, 11)]
    feature_names += [f'bm25_score_{i}' for i in range(1, 11)]
    feature_names += ['overlap_count']
    
    print(f"\nTotal features: {len(feature_names)}")
    
    train_data, temp_data = train_test_split(
        data, test_size=0.3, random_state=42,
        stratify=[d['dataset'] for d in data]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42,
        stratify=[d['dataset'] for d in temp_data]
    )
    
    print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    X_train, y_train = prepare_features(train_data, feature_names)
    X_val, y_val = prepare_features(val_data, feature_names)
    X_test, y_test = prepare_features(test_data, feature_names)
    
    print("\nTraining with hyperparameter search...")
    
    best_val_mse = float('inf')
    best_model = None
    best_params = None
    
    total_configs = len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * \
                   len(PARAM_GRID['learning_rate']) * len(PARAM_GRID['subsample'])
    
    print(f"Testing {total_configs} configurations...")
    
    config_num = 0
    for n_est in PARAM_GRID['n_estimators']:
        for max_d in PARAM_GRID['max_depth']:
            for lr in PARAM_GRID['learning_rate']:
                for subsample in PARAM_GRID['subsample']:
                    config_num += 1
                    
                    model = xgb.XGBRegressor(
                        n_estimators=n_est,
                        max_depth=max_d,
                        learning_rate=lr,
                        subsample=subsample,
                        random_state=42,
                        objective='reg:squarederror'
                    )
                    
                    model.fit(X_train, y_train, verbose=False)
                    val_pred = np.clip(model.predict(X_val), 0.0, 1.0)
                    val_mse = mean_squared_error(y_val, val_pred)
                    
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_model = model
                        best_params = {
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'learning_rate': lr,
                            'subsample': subsample
                        }
                        print(f"  Config {config_num}/{total_configs}: New best Val MSE = {val_mse:.4f}")
    
    print(f"\nBest parameters: {best_params}")
    
    print("\n=== EVALUATION ===")
    
    for split_name, X_split, y_split in [
        ('Train', X_train, y_train),
        ('Val', X_val, y_val),
        ('Test', X_test, y_test)
    ]:
        y_pred = np.clip(best_model.predict(X_split), 0.0, 1.0)
        mse = mean_squared_error(y_split, y_pred)
        r2 = r2_score(y_split, y_pred)
        
        print(f"\n{split_name} Set:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Mean α (true): {y_split.mean():.4f}")
        print(f"  Mean α (pred): {y_pred.mean():.4f}")
    
    print("\n=== TOP 10 FEATURE IMPORTANCE ===")
    
    importance = best_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    
    for name, imp in feature_importance[:10]:
        print(f"  {name}: {imp:.4f}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "xgboost_model.joblib"
    
    model_data = {
        'model': best_model,
        'feature_names': feature_names,
        'best_params': best_params
    }
    
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
