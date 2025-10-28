import argparse
import json
from pathlib import Path

from ...evaluation.metrics import compute_metrics
from ...evaluation.aggregator import aggregate_metrics
from ...shared.save_experiment_results import save_experiment_results


def get_top_k(chunks, threshold, k=5):
    """Get top-k chunks based on threshold"""
    if not chunks:
        return []
    
    max_score = max(c.get('fused_score', 0) for c in chunks)
    
    if max_score < threshold:
        sorted_chunks = sorted(chunks, key=lambda x: x.get('llm_rank', float('inf')))
    else:
        sorted_chunks = sorted(chunks, key=lambda x: x.get('fused_score', 0), reverse=True)
    
    return sorted_chunks[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='results/conditional_reranking', help='Input directory with reranked.jsonl files')
    parser.add_argument('--output-dir', default='results/conditional_reranking/evaluation', help='Output directory')
    parser.add_argument('--md-dir', default='data/converted_md', help='Markdown directory')
    parser.add_argument('--datasets', nargs='+', default=['single_file', 'single_file_multihop', 'multi_file_multihop'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    args = parser.parse_args()
    
    print("=== EVALUATION ===\n")
    
    test_data = []
    for dataset in args.datasets:
        with open(Path(args.input_dir) / f"{dataset}_reranked.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                item['dataset'] = dataset
                test_data.append(item)
    
    print(f"Loaded {len(test_data)} queries\n")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for threshold in args.thresholds:
        print(f"Threshold {threshold}...")
        
        results = []
        for i, item in enumerate(test_data):
            top = get_top_k(item['union_chunks'], threshold)
            retrieved = [{'text': c['content'], 'score': c.get('fused_score', 0)} for c in top]
            
            results.append({
                'question_id': i,
                'question': item['question'],
                'dataset': item['dataset'],
                'retrieved_chunks': retrieved,
                'ground_truth': item['ground_truth'],
                'chunk_scores': [c.get('fused_score', 0) for c in top]
            })
        
        document_cache = {}
        for r in results:
            r['metrics'] = compute_metrics(r['retrieved_chunks'], r['ground_truth'], args.md_dir, r['chunk_scores'], document_cache)
        
        results_by_dataset = {}
        for r in results:
            dataset = r['dataset']
            if dataset not in results_by_dataset:
                results_by_dataset[dataset] = []
            results_by_dataset[dataset].append(r)
        
        metrics_by_dataset = {}
        for dataset, dataset_results in results_by_dataset.items():
            metrics_by_dataset[dataset] = aggregate_metrics(dataset_results)
        
        aggregated = aggregate_metrics(results)
        
        save_experiment_results(results_by_dataset, metrics_by_dataset, output_path, f'threshold_{threshold}')
        
        all_results[threshold] = aggregated
        print(f"  T1EM: {aggregated['t1em']:.4f}, MRR: {aggregated['mrr']:.4f}\n")
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
