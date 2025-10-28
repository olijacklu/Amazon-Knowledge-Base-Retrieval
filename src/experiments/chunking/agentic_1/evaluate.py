import argparse
from pathlib import Path

from ...shared.evaluate_kb import evaluate_kb


def main():
    parser = argparse.ArgumentParser(description='Evaluate agentic_1 chunking KBs')
    parser.add_argument('--output-dir', default='results/chunking/agentic_1', help='Output directory')
    parser.add_argument('--qas-dir', default='data/qas', help='QA data directory')
    parser.add_argument('--md-dir', default='data/converted_md', help='Markdown directory')
    args = parser.parse_args()
    
    kb_ids_file = Path(args.output_dir) / 'kb_ids.json'
    evaluate_kb(kb_ids_file, args.output_dir, 'AGENTIC-1 CHUNKING', args.qas_dir, args.md_dir)


if __name__ == "__main__":
    main()
