from collections import defaultdict


def aggregate_metrics(evaluated_results):
    """Aggregate metrics across all queries"""
    if not evaluated_results:
        return {}
    
    metric_sums = defaultdict(float)
    
    for result in evaluated_results:
        metrics = result.get('metrics', {})
        for metric_name, value in metrics.items():
            metric_sums[metric_name] += value
    
    aggregated = {
        'total_questions': len(evaluated_results),
        'avg_latency': sum(r['latency'] for r in evaluated_results) / len(evaluated_results)
    }
    
    for metric_name, total in metric_sums.items():
        aggregated[metric_name] = total / len(evaluated_results)
    
    return aggregated


def print_metrics_summary(aggregated_metrics):
    """Print formatted summary of aggregated metrics"""
    print(f"\nTotal Questions: {aggregated_metrics['total_questions']}")
    print(f"Avg Latency: {aggregated_metrics['avg_latency']:.3f}s")
    
    print("\nGold Span Metrics:")
    print(f"  T1EM (Top-1 Exact Match): {aggregated_metrics.get('t1em', 0):.3f}")
    print(f"  T5EM (Top-5 Exact Match): {aggregated_metrics.get('t5em', 0):.3f}")
    print(f"  T1C (Top-1 Coverage): {aggregated_metrics.get('t1c', 0):.3f}")
    print(f"  BC (Best Coverage): {aggregated_metrics.get('bc', 0):.3f}")
    print(f"  AC (Average Coverage): {aggregated_metrics.get('ac', 0):.3f}")
    print(f"  MC (Multi-chunk Coverage): {aggregated_metrics.get('mc', 0):.3f}")
    print(f"  MRR (Mean Reciprocal Rank): {aggregated_metrics.get('mrr', 0):.3f}")
    print(f"  AT1S (Avg Top-1 Score): {aggregated_metrics.get('at1s', 0):.3f}")
    
    print("\nChunk Accuracy Metrics:")
    print(f"  T1O (Top-1 Overlap): {aggregated_metrics.get('t1o', 0):.3f}")
    print(f"  BO (Best Overlap): {aggregated_metrics.get('bo', 0):.3f}")
    print(f"  AO (Average Overlap): {aggregated_metrics.get('ao', 0):.3f}")
