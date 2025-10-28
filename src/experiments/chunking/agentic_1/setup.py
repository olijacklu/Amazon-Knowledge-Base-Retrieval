import argparse
import json
from pathlib import Path

from ..base_experiment import setup_shared_infrastructure, create_kb_with_chunking, get_chunking_name
from .config import CHUNKING_CONFIGS


def main():
    parser = argparse.ArgumentParser(description='Setup agentic_1 chunking KBs')
    parser.add_argument('--output-dir', default='results/chunking/agentic_1', help='Output directory')
    args = parser.parse_args()
    
    print("=== AGENTIC-1 CHUNKING SETUP ===")
    print(f"Creating {len(CHUNKING_CONFIGS)} Knowledge Bases")
    
    setup_shared_infrastructure()
    
    kb_ids = {}
    for i, config in enumerate(CHUNKING_CONFIGS, 1):
        name = get_chunking_name(config)
        print(f"[{i}/{len(CHUNKING_CONFIGS)}] Creating {name}...")
        kb_id = create_kb_with_chunking(config)
        kb_ids[name] = kb_id
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'kb_ids.json', 'w') as f:
        json.dump(kb_ids, f, indent=2)
    
    print(f"\nSaved KB IDs to {output_path / 'kb_ids.json'}")
    print("Setup complete!")


if __name__ == "__main__":
    main()
