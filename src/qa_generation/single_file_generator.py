import json
import time
import argparse
from pathlib import Path
from multiprocessing import Pool

from .llm_client import (
    SYSTEM_PROMPT,
    MULTIHOP_SYSTEM_PROMPT,
    build_single_hop_prompt,
    build_multihop_prompt,
    call_bedrock_claude,
    extract_json
)
from .text_processing import clean_markdown, split_into_chunks


def process_single_file(args):
    """Process a single markdown file to generate QAs and entities"""
    md_file, output_path = args
    print(f"Starting: {md_file.name}")

    content = md_file.read_text(encoding='utf-8')
    cleaned_content = clean_markdown(content)
    chunks = split_into_chunks(cleaned_content)

    qa_base_dir = output_path / md_file.stem
    qa_base_dir.mkdir(parents=True, exist_ok=True)
    
    single_qa_file = qa_base_dir / "single_hop_qas.jsonl"
    entities_file = qa_base_dir / "chunk_entities.jsonl" 
    multi_qa_file = qa_base_dir / "multi_hop_qas.jsonl"

    chunk_entities = []
    chunk_data = []

    with single_qa_file.open('w', encoding='utf-8') as single_f, entities_file.open('w', encoding='utf-8') as entities_f:
        for i, chunk in enumerate(chunks):
            try:
                response = call_bedrock_claude(SYSTEM_PROMPT, build_single_hop_prompt(chunk, max_qas=3))
                response = extract_json(response)
                
                try:
                    data = json.loads(response)
                except json.JSONDecodeError as e:
                    print(f"ERROR: {md_file.name} - Chunk {i} JSON parse failed: {e}")
                    continue

                entities = data.get("entities", [])
                chunk_entities.append(entities)
                chunk_data.append(chunk)

                entities_f.write(json.dumps({
                    "chunk": i + 1,
                    "entities": entities
                }) + "\n")

                for qa in data.get("qas", []):
                    answer = qa.get("answer", "").strip()
                    question = qa.get("question", "").strip()

                    if answer in chunk and question:
                        start_pos = chunk.find(answer)
                        end_pos = start_pos + len(answer)

                        record = {
                            "chunk": i + 1,
                            "span": {"start": start_pos, "end": end_pos},
                            "question": question,
                            "answer": answer
                        }
                        single_f.write(json.dumps(record) + "\n")
                
                time.sleep(0.5)
                        
            except Exception as e:
                print(f"FAILED: {md_file.name} - Chunk {i} exception: {e}")
    
    with multi_qa_file.open('w', encoding='utf-8') as multi_f:
        for i in range(len(chunk_entities)):
            for j in range(i + 2, len(chunk_entities)):
                if i >= len(chunk_entities) or j >= len(chunk_entities):
                    continue
                shared_entities = set(chunk_entities[i]) & set(chunk_entities[j])
                
                if shared_entities:
                    try:
                        multihop_response = call_bedrock_claude(
                            MULTIHOP_SYSTEM_PROMPT, 
                            build_multihop_prompt(chunk_data[i], chunk_data[j], max_qas=len(shared_entities))
                        )
                        
                        multihop_response = extract_json(multihop_response)
                        
                        try:
                            multihop_data = json.loads(multihop_response)
                        except json.JSONDecodeError as e:
                            print(f"ERROR: {md_file.name} - Multihop {i+1},{j+1} JSON parse failed: {e}")
                            continue
                        
                        for qa in multihop_data.get("qas", []):
                            answer = qa.get("answer", "").strip()
                            question = qa.get("question", "").strip()
                            span_info = qa.get("answer_span", {})
                            
                            passage = span_info.get("passage", "A")
                            target_chunk = chunk_data[i] if passage == "A" else chunk_data[j]
                            target_chunk_id = i + 1 if passage == "A" else j + 1
                            
                            if answer in target_chunk and question:
                                start_pos = target_chunk.find(answer)
                                end_pos = start_pos + len(answer)
                                
                                record = {
                                    "chunk_pair": [i + 1, j + 1],
                                    "answer_chunk": target_chunk_id,
                                    "span": {"start": start_pos, "end": end_pos},
                                    "question": question,
                                    "answer": answer,
                                    "shared_entities": list(shared_entities)
                                }
                                multi_f.write(json.dumps(record) + "\n")
                        
                        time.sleep(0.5)
                                
                    except Exception as e:
                        print(f"FAILED: {md_file.name} - Multihop chunks {i+1},{j+1} exception: {e}")
    
    print(f"SUCCESS: {md_file.name} - Generated QAs and entities")
    return f"Completed {md_file.name}"


def retry_failed_files(md_files, output_path, max_retries=3):
    """Retry processing failed files"""
    retry_dict = {}

    for md_file in md_files:
        try:
            qa_base_dir = output_path / md_file.stem
            single_qa_file = qa_base_dir / "single_hop_qas.jsonl"
            
            if not single_qa_file.exists():
                print(f"FAILED: {md_file.name} - No output file created")
                retry_dict[md_file] = 0
            elif single_qa_file.stat().st_size == 0:
                print(f"FAILED: {md_file.name} - Empty output file")
                retry_dict[md_file] = 0
            else:
                print(f"SUCCESS: {md_file.name} - Valid output generated")
        except Exception as e:
            print(f"FAILED: {md_file.name} - Error checking output: {e}")
            retry_dict[md_file] = 0

    while retry_dict and any(count < max_retries for count in retry_dict.values()):
        files_to_retry = [f for f, count in retry_dict.items() if count < max_retries]
        print(f"Retrying {len(files_to_retry)} failed files...")

        for md_file in files_to_retry:
            print(f"Retrying: {md_file.name} (attempt {retry_dict[md_file] + 1})")
            try:
                process_single_file((md_file, output_path))
                qa_base_dir = output_path / md_file.stem
                single_qa_file = qa_base_dir / "single_hop_qas.jsonl"
                if single_qa_file.exists() and single_qa_file.stat().st_size > 0:
                    print(f"SUCCESS: {md_file.name} - Retry successful")
                    del retry_dict[md_file]
                else:
                    print(f"FAILED: {md_file.name} - Retry {retry_dict[md_file] + 1} failed")
                    retry_dict[md_file] += 1
            except Exception as e:
                print(f"FAILED: {md_file.name} - Retry {retry_dict[md_file] + 1} error: {e}")
                retry_dict[md_file] += 1

    failed_files = [f for f, count in retry_dict.items() if count >= max_retries]
    if failed_files:
        print(f"Failed to process {len(failed_files)} files after {max_retries} retries:")
        for f in failed_files:
            print(f"  - {f.name}")


def generate_single_file_qas(input_dir, output_dir, num_workers=5):
    """Generate single-file QAs for all markdown files"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    md_files = list(input_path.glob("*.md"))
    
    with Pool(processes=num_workers) as pool:
        pool.map(process_single_file, [(f, output_path) for f in md_files])

    retry_failed_files(md_files, output_path)
    print("All files processed!")


def main():
    parser = argparse.ArgumentParser(description="Generate single-file QAs")
    parser.add_argument("--md-dir", default='data/converted_md", help="Input directory with markdown files")
    parser.add_argument("--output-dir", default='data/qas", help="Output directory for QAs")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    
    args = parser.parse_args()
    generate_single_file_qas(args.input_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
