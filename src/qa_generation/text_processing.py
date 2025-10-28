import re


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


def split_into_chunks(md_text, words_per_chunk=1000, min_words=500):
    """Split markdown text into word-based chunks"""
    words = re.findall(r"\S+", md_text)
    chunks = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i+words_per_chunk]
        if len(chunk_words) < min_words:
            continue
        chunks.append(" ".join(chunk_words))
    
    return chunks
