import json
import time
import boto3
from pathlib import Path


REGION = 'us-west-2'
BUCKET_NAME = 'arpc-converted'

bedrock_agent_client = boto3.client('bedrock-agent', region_name=REGION)
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name=REGION)
sts_client = boto3.client('sts')


def get_role_arn():
    """Get the IAM role ARN for Knowledge Base"""
    account_id = sts_client.get_caller_identity()['Account']
    return f'arn:aws:iam::{account_id}:role/space-knowledge-base'


def wait_for_kb_ready(kb_id, max_wait=300):
    """Wait for Knowledge Base to be ready"""
    print(f"Waiting for KB {kb_id} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        kb_details = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
        status = kb_details['knowledgeBase']['status']
        
        print(f"KB status: {status}")
        
        if status == 'ACTIVE':
            print("KB is ready!")
            return True
        elif status == 'FAILED':
            print("KB creation failed!")
            return False
        
        time.sleep(10)
    
    print("Timeout waiting for KB")
    return False


def retrieve_chunks(kb_id, question, k=5):
    """Retrieve top-k chunks from knowledge base"""
    start_time = time.time()
    
    response = bedrock_agent_runtime_client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={'text': question},
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': k
            }
        }
    )
    
    latency = time.time() - start_time
    
    chunks = []
    for result in response['retrievalResults']:
        chunks.append({
            'text': result['content']['text'],
            'score': result['score'],
            'location': result.get('location', {})
        })
    
    return chunks, latency


def test_retrieval(kb_ids, eval_data, k=5):
    """Test all KBs on evaluation datasets"""
    results = {}
    
    for kb_name, kb_id in kb_ids.items():
        print(f"Testing {kb_name}...")
        results[kb_name] = {}
        
        for dataset_name, questions in eval_data.items():
            print(f"  Dataset: {dataset_name}")
            dataset_results = []
            
            for i, item in enumerate(questions):
                if i % 10 == 0:
                    print(f"    Progress: {i}/{len(questions)}")
                
                chunks, latency = retrieve_chunks(kb_id, item['question'], k)
                
                result_data = {
                    'question_id': item.get('original_row_number', i),
                    'question': item['question'],
                    'ground_truth': item,
                    'retrieved_chunks': chunks,
                    'latency': latency,
                    'chunk_scores': [chunk['score'] for chunk in chunks]
                }
                
                dataset_results.append(result_data)
            
            results[kb_name][dataset_name] = dataset_results
    
    return results


def save_results(retrieval_results, metrics, output_dir):
    """Save experiment results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for kb_name, kb_results in retrieval_results.items():
        kb_dir = output_path / kb_name
        kb_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, results in kb_results.items():
            qa_results_path = kb_dir / f"{dataset_name}_qa_results.json"
            with open(qa_results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            chunks_path = kb_dir / f"{dataset_name}_chunks.json"
            chunks_data = [{'question_id': r['question_id'], 'retrieved_chunks': r['retrieved_chunks']} for r in results]
            with open(chunks_path, 'w') as f:
                json.dump(chunks_data, f, indent=2, default=str)
            
            aggregated_path = kb_dir / f"{dataset_name}_aggregated.json"
            with open(aggregated_path, 'w') as f:
                json.dump(metrics[kb_name][dataset_name], f, indent=2, default=str)
    
    print(f"Results saved to: {output_dir}")


def load_evaluation_data(qas_dir='data/qas'):
    """Load QA datasets from generated QAs"""
    qas_path = Path(qas_dir)
    
    single_hop_qas = []
    single_multihop_qas = []
    
    for paper_dir in qas_path.iterdir():
        if paper_dir.is_dir():
            single_hop_file = paper_dir / 'single_hop_qas.jsonl'
            if single_hop_file.exists():
                with open(single_hop_file, 'r') as f:
                    for line in f:
                        qa = json.loads(line)
                        qa['paper'] = paper_dir.name
                        single_hop_qas.append(qa)
            
            multi_hop_file = paper_dir / 'multi_hop_qas.jsonl'
            if multi_hop_file.exists():
                with open(multi_hop_file, 'r') as f:
                    for line in f:
                        qa = json.loads(line)
                        qa['paper'] = paper_dir.name
                        single_multihop_qas.append(qa)
    
    multi_multihop_qas = []
    multihop_dir = Path('data/multihop_qas')
    if multihop_dir.exists():
        for qa_file in multihop_dir.glob('*.jsonl'):
            with open(qa_file, 'r') as f:
                for line in f:
                    multi_multihop_qas.append(json.loads(line))
    
    eval_data = {
        'single_file': single_hop_qas,
        'single_file_multihop': single_multihop_qas,
        'multi_file_multihop': multi_multihop_qas
    }
    
    for dataset_name, data in eval_data.items():
        print(f"Loaded {len(data)} samples from {dataset_name}")
    
    return eval_data
