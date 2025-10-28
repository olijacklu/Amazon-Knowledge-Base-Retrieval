import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from .config import PARAM_GRID, FEATURE_NAMES, WH_WORDS


def extract_simple_features(question):
    """Extract 4 simple features from question"""
    words = question.lower().split()
    
    return {
        'word_count': len(words),
        'has_numbers': float(any(char.isdigit() for char in question)),
        'has_wh': float(any(word in words for word in WH_WORDS)),
        'has_acronyms': float(any(word.isupper() and len(word) > 1 for word in question.split()))
    }


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
                
                features = extract_simple_features(item['question'])
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
    parser = argparse.ArgumentParser(description='Train Ridge regression model')
    parser.add_argument('--input-dir', default='results/hybrid_retrieval', help='Input directory with merged.jsonl files')
    parser.add_argument('--output-dir', default='results/hybrid_retrieval/models', help='Output directory')
    args = parser.parse_args()
    
    print("=== RIDGE REGRESSION TRAINING ===")
    
    data = load_data(args.input_dir)
    
    feature_names = FEATURE_NAMES
    
    train_data, temp_data = train_test_split(
        data, test_size=0.3, random_state=42,
        stratify=[d['dataset'] for d in data]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42,
        stratify=[d['dataset'] for d in temp_data]
    )
    
    print(f"\nData split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    X_train, y_train = prepare_features(train_data, feature_names)
    X_val, y_val = prepare_features(val_data, feature_names)
    X_test, y_test = prepare_features(test_data, feature_names)
    
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nHyperparameter search...")
    
    grid_search = GridSearchCV(
        Ridge(random_state=42),
        PARAM_GRID,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV MSE: {-grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    print("\n=== EVALUATION ===")
    
    for split_name, X_split, y_split in [
        ('Train', X_train_scaled, y_train),
        ('Val', X_val_scaled, y_val),
        ('Test', X_test_scaled, y_test)
    ]:
        y_pred = np.clip(best_model.predict(X_split), 0.0, 1.0)
        mse = mean_squared_error(y_split, y_pred)
        r2 = r2_score(y_split, y_pred)
        
        print(f"\n{split_name} Set:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Mean α (true): {y_split.mean():.4f}")
        print(f"  Mean α (pred): {y_pred.mean():.4f}")
    
    print("\n=== FEATURE IMPORTANCE ===")
    for name, coef in zip(feature_names, best_model.coef_):
        print(f"  {name}: {coef:.4f}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "ridge_model.joblib"
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'best_params': grid_search.best_params_
    }
    
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
