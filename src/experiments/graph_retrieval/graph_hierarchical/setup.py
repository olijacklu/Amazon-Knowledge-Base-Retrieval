import argparse
import json
from pathlib import Path

from ..graph_utils import create_graph_kb
from .config import REGION, BUCKET_NAME, CHUNKING_CONFIG, GRAPH_METHOD


def main():
    parser = argparse.ArgumentParser(description='Setup graph+hierarchical KB')
    parser.add_argument('--output-dir', default='results/graph_retrieval/graph_hierarchical', help='Output directory')
    args = parser.parse_args()
    
    print("=== GRAPH + HIERARCHICAL SETUP ===")
    print(f"Method: Neptune Analytics + {GRAPH_METHOD}")
    
    print("Creating Knowledge Base with Neptune Analytics...")
    kb_id = create_graph_kb(CHUNKING_CONFIG, BUCKET_NAME, REGION)
    kb_name = f"graph-{GRAPH_METHOD}"
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'kb_ids.json', 'w') as f:
        json.dump({kb_name: kb_id}, f, indent=2)
    
    print(f"\nSaved KB ID to {output_path / 'kb_ids.json'}")
    print("Setup complete!")


if __name__ == "__main__":
    main()
