import re
import argparse
from pathlib import Path


def clean_markdown(md_text):
    """Remove image references and following blank lines from markdown text"""
    lines = md_text.split('\n')
    cleaned_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        if re.search(r'!\[.*?\]\(.*?\)', line):
            i += 1
            if i < len(lines) and lines[i].strip() == '':
                i += 1
        else:
            cleaned_lines.append(line)
            i += 1
    
    return '\n'.join(cleaned_lines)


def clean_markdown_files(md_dir):
    """Clean all markdown files in a directory"""
    md_dir = Path(md_dir)
    
    if not md_dir.exists():
        print(f"Error: Directory {md_dir} does not exist")
        return
    
    md_files = list(md_dir.glob("*.md"))
    
    if not md_files:
        print(f"No markdown files found in {md_dir}")
        return
    
    for md_file in md_files:
        print(f"Cleaning {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = clean_markdown(content)
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
    
    print(f"Cleaned {len(md_files)} markdown files!")


def main():
    parser = argparse.ArgumentParser(description="Clean markdown files")
    parser.add_argument("--md-dir", default='data/converted_md", help="Directory with markdown files")
    
    args = parser.parse_args()
    clean_markdown_files(args.md_dir)


if __name__ == "__main__":
    main()
