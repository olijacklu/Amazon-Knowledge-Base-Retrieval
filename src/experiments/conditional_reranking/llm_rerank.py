import argparse
import json
import time
import boto3
from pathlib import Path


MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
FALLBACK_MODEL_ID = "us.anthropic.claude-3-haiku-20240307-v1:0"
REGION = "us-west-2"
MAX_TOKENS = 1000
TEMPERATURE = 0.2


def invoke_claude_haiku(prompt, max_retries=5):
    """Invoke Claude Haiku with retry logic"""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)

    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
    })

    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                contentType='application/json',
                accept='application/json',
                body=request_body
            )

            result = ""
            for event in response.get('body'):
                chunk = json.loads(event['chunk']['bytes'].decode())
                if chunk['type'] == 'content_block_delta':
                    result += chunk['delta']['text']
                elif chunk['type'] == 'message_delta':
                    if 'stop_reason' in chunk['delta']:
                        break
            
            return result.strip()
            
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                if attempt == 0:
                    print(f"  Throttling detected, trying fallback model")
                    fallback_request = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": MAX_TOKENS,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                    })
                    try:
                        fallback_response = bedrock_runtime.invoke_model_with_response_stream(
                            modelId=FALLBACK_MODEL_ID,
                            contentType='application/json',
                            accept='application/json',
                            body=fallback_request
                        )
                        result = ""
                        for event in fallback_response.get('body'):
                            chunk = json.loads(event['chunk']['bytes'].decode())
                            if chunk['type'] == 'content_block_delta':
                                result += chunk['delta']['text']
                            elif chunk['type'] == 'message_delta':
                                if 'stop_reason' in chunk['delta']:
                                    break
                        return result.strip()
                    except:
                        pass
                
                print(f"  Token limit hit, waiting 30s before retry {attempt + 1}/{max_retries}")
                time.sleep(30)
            else:
                print(f"  Claude Haiku error: {e}")
                return ""
    
    return ""


def create_rerank_prompt(question, chunks):
    """Create reranking prompt"""
    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        chunks_text += f"\n{i}. {chunk['content']}\n"
    
    prompt = f"""Given the following query and retrieved chunks (ordered by highest score to lowest), rank the chunks by their relevance to answering the query.

Query: {question}

Chunks:{chunks_text}

Please analyze each chunk and return ONLY a JSON array of document numbers ranked from most relevant to least relevant.
Example format: [1, 5, 3, 7, 2, ...]

Return ONLY the JSON array, nothing else."""
    
    return prompt


def parse_rerank_response(response, num_chunks):
    """Parse LLM response to extract ranking"""
    try:
        response = response.strip()
        if response.startswith('[') and response.endswith(']'):
            ranking = json.loads(response)
            if len(ranking) == num_chunks and all(1 <= r <= num_chunks for r in ranking):
                return ranking
    except:
        pass
    
    return list(range(1, num_chunks + 1))


def rerank_chunks(question, chunks):
    """Rerank chunks using LLM"""
    if len(chunks) == 0:
        return chunks
    
    sorted_chunks = sorted(chunks, key=lambda x: x.get('fused_score', 0), reverse=True)
    
    prompt = create_rerank_prompt(question, sorted_chunks)
    
    response = invoke_claude_haiku(prompt)
    
    if not response:
        return chunks
    
    ranking = parse_rerank_response(response, len(sorted_chunks))
    
    reranked_chunks = [sorted_chunks[r - 1] for r in ranking]
    
    for i, chunk in enumerate(reranked_chunks, 1):
        chunk['llm_rank'] = i
    
    return reranked_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='results/conditional_reranking', help='Input directory with with_scores.jsonl files')
    parser.add_argument('--output-dir', default='results/conditional_reranking', help='Output directory')
    parser.add_argument('--datasets', nargs='+', default=['single_file', 'single_file_multihop', 'multi_file_multihop'])
    args = parser.parse_args()
    
    print("=== LLM RERANKING ===")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dataset in args.datasets:
        input_file = Path(args.input_dir) / f"{dataset}_with_scores.jsonl"
        output_file = output_path / f"{dataset}_reranked.jsonl"
        
        print(f"\nProcessing {dataset}...")
        
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        results = []
        for i, line in enumerate(lines, 1):
            if i % 100 == 0:
                print(f"  {i}/{len(lines)}")
            
            item = json.loads(line.strip())
            reranked = rerank_chunks(item['question'], item['union_chunks'])
            
            results.append({
                'question': item['question'],
                'ground_truth': item['ground_truth'],
                'union_chunks': reranked
            })
        
        with open(output_file, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        
        print(f"  Saved {len(results)} queries")


if __name__ == "__main__":
    main()
