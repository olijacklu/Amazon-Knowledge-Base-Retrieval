import argparse
import json
import math
from pathlib import Path
from sklearn.model_selection import train_test_split

from ...evaluation.metrics import compute_metrics


def wilson_lower_bound(k, n):
    """Wilson score interval lower bound"""
    if n == 0:
        return 0.0
    p_hat = k / n
    z = 1.96
    denominator = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (center - margin) / denominator


def check_top5_correct(chunks, ground_truth, ranking_key, md_dir):
    """Check if top-5 contains answer"""
    if ranking_key == 'fused_score':
        sorted_chunks = sorted(chunks, key=lambda x: x.get('fused_score', 0), reverse=True)
    else:
        sorted_chunks = sorted(chunks, key=lambda x: x.get('llm_rank', float('inf')))
    
    top5 = [{'text': c['content'], 'score': c.get('fused_score', 0)} for c in sorted_chunks[:5]]
    metrics = compute_metrics(top5, ground_truth, md_dir)
    return metrics['bc'] > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='results/conditional_reranking', help='Input directory with reranked.jsonl files')
    parser.add_argument('--output-dir', default='results/conditional_reranking/calibration', help='Output directory')
    parser.add_argument('--md-dir', default='data/converted_md', help='Markdown directory')
    parser.add_argument('--datasets', nargs='+', default=['single_file', 'single_file_multihop', 'multi_file_multihop'])
    parser.add_argument('--target-precision', type=float, default=0.5)
    parser.add_argument('--min-samples', type=int, default=30)
    args = parser.parse_args()
    
    print("=== THRESHOLD CALIBRATION ===\n")
    
    all_data = []
    for dataset in args.datasets:
        with open(Path(args.input_dir) / f"{dataset}_reranked.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                item['dataset'] = dataset
                all_data.append(item)
    
    print(f"Loaded {len(all_data)} queries")
    
    cal_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=[d['dataset'] for d in all_data])
    print(f"Calibration: {len(cal_data)}, Test: {len(test_data)}\n")
    
    features = []
    for item in cal_data:
        fused_top1 = max(c.get('fused_score', 0) for c in item['union_chunks'])
        features.append({
            'fused_top1_score': fused_top1,
            'is_correct_fusion': check_top5_correct(item['union_chunks'], item['ground_truth'], 'fused_score', args.md_dir),
            'is_correct_llm': check_top5_correct(item['union_chunks'], item['ground_truth'], 'llm_rank', args.md_dir)
        })
    
    thresholds = [i * 0.05 for i in range(21)]
    results = []
    
    for tau in thresholds:
        skip = [f for f in features if f['fused_top1_score'] >= tau]
        n = len(skip)
        k = sum(1 for f in skip if f['is_correct_fusion'])
        
        if n == 0:
            continue
        
        precision = k / n
        lower = wilson_lower_bound(k, n)
        meets = lower >= args.target_precision and n >= args.min_samples
        
        results.append({'tau': tau, 'n': n, 'k': k, 'precision': precision, 'lower': lower, 'meets': meets})
        print(f"{'✓' if meets else '✗'} τ={tau:.2f}: n={n:4d}, k={k:4d}, p={precision:.3f}, CI={lower:.3f}")
    
    valid = [r for r in results if r['meets']]
    optimal_tau = min(valid, key=lambda x: x['tau'])['tau'] if valid else None
    
    if optimal_tau:
        print(f"\nOptimal: τ={optimal_tau:.2f}")
    else:
        print(f"\nNo valid threshold found")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "calibration_results.json", 'w') as f:
        json.dump({
            'calibration_set_size': len(cal_data),
            'target_precision': args.target_precision,
            'min_samples': args.min_samples,
            'optimal_tau': optimal_tau,
            'candidate_thresholds': results
        }, f, indent=2)
    
    if optimal_tau is not None:
        print(f"\nVerifying on test set...")
        test_features = []
        for item in test_data:
            fused_top1 = max(c.get('fused_score', 0) for c in item['union_chunks'])
            test_features.append({
                'fused_top1_score': fused_top1,
                'is_correct_fusion': check_top5_correct(item['union_chunks'], item['ground_truth'], 'fused_score', args.md_dir),
                'is_correct_llm': check_top5_correct(item['union_chunks'], item['ground_truth'], 'llm_rank', args.md_dir)
            })
        
        skip_group = [f for f in test_features if f['fused_top1_score'] >= optimal_tau]
        rerank_group = [f for f in test_features if f['fused_top1_score'] < optimal_tau]
        
        skip_n = len(skip_group)
        skip_correct = sum(1 for f in skip_group if f['is_correct_fusion'])
        skip_precision = skip_correct / skip_n if skip_n > 0 else 0
        
        rerank_n = len(rerank_group)
        rerank_correct_fusion = sum(1 for f in rerank_group if f['is_correct_fusion'])
        rerank_correct_llm = sum(1 for f in rerank_group if f['is_correct_llm'])
        
        print(f"\n{'=' * 80}")
        print("TEST SET RESULTS")
        print(f"{'=' * 80}")
        print(f"\nSkip group (fused_score >= {optimal_tau:.2f}):")
        print(f"  Count: {skip_n}/{len(test_features)} ({skip_n/len(test_features)*100:.1f}%)")
        print(f"  Precision (fusion top-5): {skip_precision:.3f}")
        
        print(f"\nRerank group (fused_score < {optimal_tau:.2f}):")
        print(f"  Count: {rerank_n}/{len(test_features)} ({rerank_n/len(test_features)*100:.1f}%)")
        if rerank_n > 0:
            print(f"  Precision (fusion top-5): {rerank_correct_fusion/rerank_n:.3f}")
            print(f"  Precision (LLM top-5): {rerank_correct_llm/rerank_n:.3f}")
            improvement = (rerank_correct_llm - rerank_correct_fusion) / rerank_n
            print(f"  LLM improvement: {improvement:+.3f}")
        
        print(f"\nOverall:")
        total_correct = skip_correct + rerank_correct_llm
        overall_precision = total_correct / len(test_features)
        always_fusion = sum(1 for f in test_features if f['is_correct_fusion']) / len(test_features)
        always_llm = sum(1 for f in test_features if f['is_correct_llm']) / len(test_features)
        print(f"  Precision (selective strategy): {overall_precision:.3f}")
        print(f"  Precision (always fusion): {always_fusion:.3f}")
        print(f"  Precision (always LLM): {always_llm:.3f}")
        print(f"{'=' * 80}\n")
        
        with open(output_path / "test_results.json", 'w') as f:
            json.dump({
                'tau': optimal_tau,
                'test_set_size': len(test_features),
                'skip_group': {
                    'count': skip_n,
                    'percentage': skip_n / len(test_features) * 100,
                    'precision_fusion': skip_precision
                },
                'rerank_group': {
                    'count': rerank_n,
                    'percentage': rerank_n / len(test_features) * 100,
                    'precision_fusion': rerank_correct_fusion / rerank_n if rerank_n > 0 else 0,
                    'precision_llm': rerank_correct_llm / rerank_n if rerank_n > 0 else 0,
                    'llm_improvement': (rerank_correct_llm - rerank_correct_fusion) / rerank_n if rerank_n > 0 else 0
                },
                'overall': {
                    'precision_selective': overall_precision,
                    'precision_always_fusion': always_fusion,
                    'precision_always_llm': always_llm
                }
            }, f, indent=2)


if __name__ == "__main__":
    main()
