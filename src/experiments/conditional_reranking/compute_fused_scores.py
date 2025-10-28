import argparse
import json
import numpy as np
import joblib
from pathlib import Path

from ..hybrid_retrieval.xgboost.config import WH_WORDS


def extract_xgboost_features(question, union_chunks):
    """Extract 25 features for XGBoost"""
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


def predict_alpha(model_data, question, union_chunks):
    """Predict alpha using XGBoost model"""
    features = extract_xgboost_features(question, union_chunks)
    feature_names = model_data['feature_names']
    X = np.array([[features[name] for name in feature_names]])
    alpha = model_data['model'].predict(X)[0]
    return np.clip(alpha, 0.0, 1.0)


def compute_fused_scores(union_chunks, alpha):
    """Compute fused scores and add to chunks"""
    alpha = float(alpha)
    for chunk in union_chunks:
        fused_score = alpha * chunk['normalized_semantic_score'] + (1 - alpha) * chunk['normalized_bm25_score']
        chunk['fused_score'] = float(fused_score)
        chunk['predicted_alpha'] = alpha
    return union_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='results/hybrid_retrieval', help='Input directory with merged.jsonl files')
    parser.add_argument('--model-path', default='results/hybrid_retrieval/models/xgboost_model.joblib', help='Path to trained model')
    parser.add_argument('--output-dir', default='results/conditional_reranking', help='Output directory')
    parser.add_argument('--datasets', nargs='+', default=['single_file', 'single_file_multihop', 'multi_file_multihop'])
    args = parser.parse_args()
    
    print("=== COMPUTING FUSED SCORES ===")
    
    model_data = joblib.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dataset in args.datasets:
        input_file = Path(args.input_dir) / f"{dataset}_merged.jsonl"
        output_file = output_path / f"{dataset}_with_scores.jsonl"
        
        print(f"\nProcessing {dataset}...")
        
        results = []
        with open(input_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                alpha = predict_alpha(model_data, item['question'], item['union_chunks'])
                chunks_with_scores = compute_fused_scores(item['union_chunks'], alpha)
                
                results.append({
                    'question': item['question'],
                    'ground_truth': item['ground_truth'],
                    'union_chunks': chunks_with_scores
                })
        
        with open(output_file, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        
        print(f"  Saved {len(results)} queries")


if __name__ == "__main__":
    main()
