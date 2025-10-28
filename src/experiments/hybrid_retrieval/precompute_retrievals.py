import argparse
import boto3
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
from difflib import SequenceMatcher

from ..shared.kb_utils import load_evaluation_data


REGION = 'us-west-2'


def extract_chunks_from_s3(s3_path):
    """Extract chunks from S3 KB intermediate storage"""
    s3_client = boto3.client('s3', region_name=REGION)
    
    if not s3_path.startswith('s3://'):
        raise ValueError("S3 path must start with s3://")
    
    path_parts = s3_path[5:].split('/', 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    chunks = []
    
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith('.JSON'):
                continue
                
            try:
                response = s3_client.get_object(Bucket=bucket, Key=key)
                chunk_data = response['Body'].read().decode('utf-8')
                
                for content in chunk_data.get('fileContents', []):
                    chunks.append({
                        'chunk_id': f"{key}_{len(chunks)}",
                        'content': content.get('contentBody', ''),
                        'metadata': content.get('contentMetadata', {}),
                        'source_file': key
                    })
                    
            except Exception as e:
                print(f"Error processing {key}: {e}")
                continue
    
    print(f"Extracted {len(chunks)} chunks from S3")
    return chunks


def tokenize_text(text):
    """Simple tokenization for BM25"""
    return text.lower().split()


def build_bm25_index(chunks):
    """Build BM25 index from chunks"""
    print(f"Building BM25 index from {len(chunks)} chunks...")
    
    tokenized_corpus = [tokenize_text(chunk['content']) for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    print("BM25 index created")
    return bm25_index


def get_embedding(text, bedrock_runtime):
    """Get Titan v2 embedding for text"""
    request_body = {
        "inputText": text,
        "dimensions": 1024,
        "normalize": True
    }
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType='application/json',
            accept='application/json',
            body=str(request_body)
        )
        response_body = response['body'].read().decode('utf-8')
        return np.array(response_body['embedding'])
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(1024)


def precompute_chunk_embeddings(chunks):
    """Precompute semantic embeddings for all chunks"""
    print(f"Precomputing embeddings for {len(chunks)} chunks...")
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(chunks)}")
        
        embedding = get_embedding(chunk['content'], bedrock_runtime)
        embeddings.append(embedding)
    
    print("Embeddings precomputed")
    return np.array(embeddings)


def retrieve_and_merge(query, answer, chunk_embeddings, bm25_index, chunks, k=10):
    """Retrieve top-k from both methods, merge, compute all scores and LSS"""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)
    
    query_embedding = get_embedding(query, bedrock_runtime)
    semantic_similarities = np.dot(chunk_embeddings, query_embedding)
    
    tokenized_query = tokenize_text(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    semantic_top_indices = set(np.argsort(semantic_similarities)[-k:][::-1])
    bm25_top_indices = set(np.argsort(bm25_scores)[-k:][::-1])
    
    union_indices = semantic_top_indices | bm25_top_indices
    
    max_semantic = max(semantic_similarities[idx] for idx in union_indices)
    max_bm25 = max(bm25_scores[idx] for idx in union_indices) if max(bm25_scores[idx] for idx in union_indices) > 0 else 1.0
    
    union_chunks = []
    for idx in union_indices:
        chunk = chunks[idx]
        semantic_score = float(semantic_similarities[idx])
        bm25_score = float(bm25_scores[idx])
        
        normalized_semantic = (semantic_score + 1) / (max_semantic + 1)
        normalized_bm25 = bm25_score / max_bm25
        
        answer_words = answer.lower().split()
        chunk_words = chunk['content'].lower().split()
        matcher = SequenceMatcher(None, answer_words, chunk_words)
        matches = matcher.get_matching_blocks()
        matched_answer_words = sum(match.size for match in matches[:-1])
        lss = matched_answer_words / len(answer_words) if answer_words else 0.0
        
        union_chunks.append({
            'chunk_id': chunk['chunk_id'],
            'content': chunk['content'],
            'semantic_score': semantic_score,
            'normalized_semantic_score': normalized_semantic,
            'bm25_score': bm25_score,
            'normalized_bm25_score': normalized_bm25,
            'lss': lss
        })
    
    return union_chunks


def main():
    parser = argparse.ArgumentParser(description='Precompute dual retrievals')
    parser.add_argument('--s3-path', required=True, help='S3 path to KB chunks')
    parser.add_argument('--output-dir', default='results/hybrid_retrieval', help='Output directory')
    args = parser.parse_args()
    
    print("=== PRECOMPUTING DUAL RETRIEVALS ===")
    
    print("Extracting chunks from S3...")
    chunks = extract_chunks_from_s3(args.s3_path)
    
    print("Building BM25 index...")
    bm25_index = build_bm25_index(chunks)
    
    print("Precomputing semantic embeddings...")
    chunk_embeddings = precompute_chunk_embeddings(chunks)
    
    print("Loading evaluation data...")
    eval_data = load_evaluation_data('data/qas')
    
    print("Running dual retrieval and merging...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, questions in eval_data.items():
        print(f"Processing {dataset_name} ({len(questions)} questions)...")
        
        results = []
        
        for i, item in enumerate(questions):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(questions)}")
            
            query = item['question']
            answer = item.get('answer', '')
            
            union_chunks = retrieve_and_merge(query, answer, chunk_embeddings, bm25_index, chunks, k=10)
            
            results.append({
                'question_id': item.get('original_row_number', i),
                'question': query,
                'ground_truth': item,
                'union_chunks': union_chunks
            })
        
        output_file = output_path / f"{dataset_name}_merged.jsonl"
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(str(result) + '\n')
        
        print(f"Saved {len(results)} merged results to {output_file}")
    
    print("Precomputation and merging complete!")


if __name__ == "__main__":
    main()
