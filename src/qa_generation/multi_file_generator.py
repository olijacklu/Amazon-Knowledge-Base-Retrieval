import json
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

from .llm_client import (
    MULTIHOP_SYSTEM_PROMPT,
    build_multihop_prompt,
    call_bedrock_claude,
    extract_json
)
from .text_processing import clean_markdown, split_into_chunks


def load_paper_entities(qas_dir, tags_to_papers):
    """Load entities for each paper from chunk_entities.jsonl files"""
    paper_entities = {}
    paper_to_tags = {}
    
    for tag, papers in tags_to_papers.items():
        for paper in papers:
            title = paper['title']
            paper_to_tags[title] = paper_to_tags.get(title, set())
            paper_to_tags[title].add(tag)
            
            folder_name = title.replace(' ', '_')
            entities_file = Path(qas_dir) / folder_name / 'chunk_entities.jsonl'
            
            if entities_file.exists():
                entities_set = set()
                with open(entities_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and line.startswith('{') and line.endswith('}'):
                            try:
                                chunk_data = json.loads(line)
                                if 'entities' in chunk_data:
                                    entities_set.update(chunk_data['entities'])
                            except json.JSONDecodeError:
                                continue
                paper_entities[title] = entities_set
    
    return paper_entities, paper_to_tags


def find_entity_pairs(paper_entities, paper_to_tags):
    """Find pairs of papers that share both tags and entities"""
    entity_pairs = defaultdict(list)
    
    papers = list(paper_entities.keys())
    for i in range(len(papers)):
        for j in range(i+1, len(papers)):
            paper1, paper2 = papers[i], papers[j]
            
            shared_tags = paper_to_tags[paper1] & paper_to_tags[paper2]
            if shared_tags:
                shared_entities = paper_entities[paper1] & paper_entities[paper2]
                for entity in shared_entities:
                    entity_pairs[entity].append((paper1, paper2))
    
    return entity_pairs


def find_entity_chunk(entity, paper_title, qas_dir):
    """Find the first chunk containing a specific entity for a paper"""
    folder_name = paper_title.replace(' ', '_')
    entities_file = Path(qas_dir) / folder_name / 'chunk_entities.jsonl'
    
    if not entities_file.exists():
        return None
    
    with open(entities_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('{') and line.endswith('}'):
                try:
                    chunk_data = json.loads(line)
                    if 'entities' in chunk_data and entity in chunk_data['entities']:
                        return chunk_data['chunk']
                except json.JSONDecodeError:
                    continue
    
    return None


def process_entity_pair(args):
    """Process a single entity pair to generate cross-paper multi-hop QAs"""
    entity, selected_pair, chunks, md_dir, output_dir = args
    paper1, paper2 = selected_pair
    
    print(f"Starting: {entity}")
    
    md_file1 = md_dir / f"{paper1.replace(' ', '_')}.md"
    md_file2 = md_dir / f"{paper2.replace(' ', '_')}.md"
    
    if not (md_file1.exists() and md_file2.exists()):
        print(f"FAILED: {entity} - Missing markdown files")
        return f"FAILED: {entity} - Missing markdown files"
        
    with open(md_file1, 'r', encoding='utf-8') as f:
        content1 = clean_markdown(f.read())
        chunks1 = split_into_chunks(content1)
    
    with open(md_file2, 'r', encoding='utf-8') as f:
        content2 = clean_markdown(f.read())
        chunks2 = split_into_chunks(content2)
    
    if chunks[0] > len(chunks1) or chunks[1] > len(chunks2):
        print(f"FAILED: {entity} - Invalid chunk numbers")
        return f"FAILED: {entity} - Invalid chunk numbers"
        
    chunk1 = chunks1[chunks[0] - 1]
    chunk2 = chunks2[chunks[1] - 1]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = call_bedrock_claude(
                MULTIHOP_SYSTEM_PROMPT,
                build_multihop_prompt(chunk1, chunk2, max_qas=1, paper_a=paper1, paper_b=paper2)
            )
            
            response = extract_json(response)
            
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"ERROR: {entity} - Attempt {attempt + 1} JSON parse failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
            
            for qa in data.get("qas", []):
                answer = qa.get("answer", "").strip()
                question = qa.get("question", "").strip()
                span_info = qa.get("answer_span", {})
                
                if not (answer and question):
                    continue
                
                target_chunk = None
                target_paper = None
                target_chunk_num = None
                
                passage = span_info.get("passage", "A")
                if passage == "A" and answer in chunk1:
                    target_chunk = chunk1
                    target_paper = paper1
                    target_chunk_num = chunks[0]
                elif passage == "B" and answer in chunk2:
                    target_chunk = chunk2
                    target_paper = paper2
                    target_chunk_num = chunks[1]
                elif answer in chunk1:
                    target_chunk = chunk1
                    target_paper = paper1
                    target_chunk_num = chunks[0]
                elif answer in chunk2:
                    target_chunk = chunk2
                    target_paper = paper2
                    target_chunk_num = chunks[1]
                
                if target_chunk:
                    start_pos = target_chunk.find(answer)
                    end_pos = start_pos + len(answer)
                    
                    record = {
                        "entity": entity,
                        "paper_pair": [paper1, paper2],
                        "chunk_pair": chunks,
                        "answer_paper": target_paper,
                        "answer_chunk": target_chunk_num,
                        "span": {"start": start_pos, "end": end_pos},
                        "question": question,
                        "answer": answer
                    }
                    
                    output_file = output_dir / f"{entity.replace(' ', '_')}.jsonl"
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record) + "\n")
                    
                    print(f"SUCCESS: {entity} - Generated QA")
                    return f"SUCCESS: Generated QA for entity: {entity}"
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"FAILED: {entity} - Attempt {attempt + 1} exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    print(f"FAILED: {entity} after {max_retries} attempts")
    return f"FAILED: Entity {entity} after {max_retries} attempts"


def generate_multi_file_qas(md_dir, qas_dir, tags_file, output_dir, num_workers=5):
    """Generate multi-file QAs for entity pairs across papers"""
    md_path = Path(md_dir)
    qas_path = Path(qas_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading tags to papers mapping...")
    with open(tags_file, 'r') as f:
        tags_to_papers = json.load(f)

    print("Step 1: Bundling entities per paper into sets...")
    paper_entities, paper_to_tags = load_paper_entities(qas_path, tags_to_papers)
    print(f"Loaded {len(paper_entities)} papers with entities")

    print("Step 2: Finding pairs sharing tag and entity...")
    entity_pairs = find_entity_pairs(paper_entities, paper_to_tags)
    print(f"Found {len(entity_pairs)} entities with shared pairs")

    print("Step 3: Preparing tasks for parallel processing...")
    tasks = []
    for entity, pairs in entity_pairs.items():
        if len(pairs) == 1:
            selected_pair = random.choice(pairs)
            
            chunks = []
            for paper_title in selected_pair:
                chunk_num = find_entity_chunk(entity, paper_title, qas_path)
                chunks.append(chunk_num)
            
            if None not in chunks:
                tasks.append((entity, selected_pair, chunks, md_path, output_path))

    print(f"Total entities with >1 valid pairs: {len(tasks)}")
    print("Selected pairs for processing:")
    for entity, selected_pair, chunks, _, _ in tasks:
        print(f"Entity: {entity}")
        print(f"  Pair: {selected_pair}")
        print(f"  Chunks: {chunks}")
        print("---")

    print("Starting parallel processing...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_entity_pair, tasks)

    print("\nFinal Results:")
    success_count = sum(1 for r in results if "SUCCESS" in r)
    failed_count = sum(1 for r in results if "FAILED" in r)
    
    for result in results:
        print(result)

    print(f"\nSUMMARY:")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print("Multi-hop QA generation completed!")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-file QAs")
    parser.add_argument("--md-dir", default='data/converted_md", help="Directory with markdown files")
    parser.add_argument("--qas-dir", default='data/qas", help="Directory with single-file QAs")
    parser.add_argument("--tags-file", required=True, help="Path to tags_to_papers.json")
    parser.add_argument("--output-dir", default='data/multihop_qas", help="Output directory")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    
    args = parser.parse_args()
    generate_multi_file_qas(args.md_dir, args.qas_dir, args.tags_file, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
