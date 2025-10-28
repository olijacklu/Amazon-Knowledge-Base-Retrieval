import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


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


def find_entity_chunks(entity, paper_title, qas_dir):
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


def analyze_entity_pairs(tags_file, qas_dir, output_file=None):
    """Analyze entity pairs across papers and optionally save results"""
    # Load tags to papers mapping
    with open(tags_file, 'r') as f:
        tags_to_papers = json.load(f)
    
    # Load paper entities
    paper_entities, paper_to_tags = load_paper_entities(qas_dir, tags_to_papers)
    
    # Find entity pairs
    entity_pairs = find_entity_pairs(paper_entities, paper_to_tags)
    
    # Analyze pairs with multiple occurrences
    results = []
    for entity, pairs in entity_pairs.items():
        if len(pairs) > 1:
            selected_pair = random.choice(pairs)
            
            chunks = []
            for paper_title in selected_pair:
                chunk = find_entity_chunks(entity, paper_title, qas_dir)
                chunks.append(chunk)
            
            result = {
                'entity': entity,
                'pair': selected_pair,
                'chunks': chunks
            }
            results.append(result)
            
            print(f"Entity: {entity}")
            print(f"Pair: {selected_pair}")
            print(f"Chunks: {chunks}")
            print("---")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze entity pairs")
    parser.add_argument("--tags-file", required=True, help="Path to tags_to_papers.json")
    parser.add_argument("--qas-dir", required=True, help="Directory containing QA folders")
    parser.add_argument("--output-file", help="Output JSON file for results")
    
    args = parser.parse_args()
    analyze_entity_pairs(args.tags_file, args.qas_dir, args.output_dir)


if __name__ == "__main__":
    main()
