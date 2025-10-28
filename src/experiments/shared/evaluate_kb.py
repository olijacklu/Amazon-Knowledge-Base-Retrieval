import json

from .kb_utils import test_retrieval, save_results, load_evaluation_data
from ...evaluation.metrics import compute_metrics
from ...evaluation.aggregator import aggregate_metrics


def evaluate_kb(kb_ids_file, output_dir, experiment_name, qas_dir='data/qas', md_dir='data/converted_md'):
    """Shared evaluation logic for all KB experiments"""
    print(f"=== {experiment_name} EVALUATION ===")
    
    with open(kb_ids_file, 'r') as f:
        kb_ids = json.load(f)
    
    print(f"Loaded {len(kb_ids)} KB(s)")
    
    eval_data = load_evaluation_data(qas_dir)
    
    print("Running retrieval tests...")
    retrieval_results = test_retrieval(kb_ids, eval_data, k=5)
    
    print("Computing metrics...")
    document_cache = {}
    all_metrics = {}
    for kb_name, kb_results in retrieval_results.items():
        all_metrics[kb_name] = {}
        for dataset_name, results in kb_results.items():
            for result in results:
                result['metrics'] = compute_metrics(
                    result['retrieved_chunks'],
                    result['ground_truth'],
                    md_dir,
                    result.get('chunk_scores'),
                    document_cache
                )
            metrics = aggregate_metrics(results)
            all_metrics[kb_name][dataset_name] = metrics
    
    save_results(retrieval_results, all_metrics, output_dir)
    
    print("\n=== RESULTS SUMMARY ===")
    for kb_name, kb_metrics in all_metrics.items():
        print(f"\n{kb_name}:")
        for dataset_name, metrics in kb_metrics.items():
            print(f"  {dataset_name}: T1EM={metrics['t1em']:.4f}, MRR={metrics['mrr']:.4f}, BC={metrics['bc']:.4f}")
    
    print("\nEvaluation complete!")
