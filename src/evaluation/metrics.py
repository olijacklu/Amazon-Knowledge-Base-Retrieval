from difflib import SequenceMatcher


def calculate_text_overlap(text1, text2):
    """Calculate bidirectional text overlap using word-level sequence matching"""
    if not text1 or not text2:
        return 0
    
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    matcher = SequenceMatcher(None, words1, words2)
    return matcher.ratio()


def calculate_answer_coverage(chunk_text, answer):
    """Calculate unidirectional coverage: proportion of answer words found in chunk via LCS"""
    if not answer or not chunk_text:
        return 0
    
    answer_words = answer.lower().split()
    chunk_words = chunk_text.lower().split()
    
    matcher = SequenceMatcher(None, answer_words, chunk_words)
    matches = matcher.get_matching_blocks()
    
    matched_answer_words = sum(match.size for match in matches[:-1])
    return matched_answer_words / len(answer_words) if answer_words else 0


def gold_span_metrics(retrieved_chunks, ground_truth, chunk_scores=None):
    """Calculate all gold span accuracy metrics"""
    answer = ground_truth['answer']
    
    if not retrieved_chunks:
        return {
            't1em': 0,
            't5em': 0,
            't1c': 0,
            'bc': 0,
            'ac': 0,
            'mc': 0,
            'mrr': 0,
            'at1s': 0
        }
    
    # T1EM: Top-1 Exact Match
    t1em = 1 if answer.lower() in retrieved_chunks[0]['text'].lower() else 0
    
    # T5EM: Top-5 Exact Match
    t5em = 0
    for chunk in retrieved_chunks[:5]:
        if answer.lower() in chunk['text'].lower():
            t5em = 1
            break
    
    # T1C: Top-1 Coverage
    t1c = calculate_answer_coverage(retrieved_chunks[0]['text'], answer)
    
    # BC: Best Coverage
    coverages = [calculate_answer_coverage(chunk['text'], answer) for chunk in retrieved_chunks]
    bc = max(coverages) if coverages else 0
    
    # AC: Average Coverage
    ac = sum(coverages) / len(coverages) if coverages else 0
    
    # MC: Multi-chunk Coverage
    combined_text = " ".join([chunk['text'] for chunk in retrieved_chunks])
    mc = calculate_answer_coverage(combined_text, answer)
    
    # MRR: Mean Reciprocal Rank
    answer_rank = 0
    for i, chunk in enumerate(retrieved_chunks):
        if answer.lower() in chunk['text'].lower():
            answer_rank = i + 1
            break
    mrr = 1.0 / answer_rank if answer_rank > 0 else 0.0
    
    # AT1S: Average Top-1 Score
    at1s = chunk_scores[0] if chunk_scores and len(chunk_scores) > 0 else 0
    
    return {
        't1em': t1em,
        't5em': t5em,
        't1c': t1c,
        'bc': bc,
        'ac': ac,
        'mc': mc,
        'mrr': mrr,
        'at1s': at1s
    }


def load_document(paper_name, md_dir, document_cache):
    """Load document from file and cache it"""
    if paper_name not in document_cache:
        file_path = f"{md_dir}/{paper_name.replace(' ', '_')}.md"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document_cache[paper_name] = f.read()
        except FileNotFoundError:
            print(f"Warning: Could not find {file_path}")
            document_cache[paper_name] = ""
    return document_cache[paper_name]


def get_ground_truth_texts(ground_truth, md_dir, document_cache=None):
    """Extract actual ground truth text from original documents"""
    if document_cache is None:
        document_cache = {}
    
    if 'chunk_span' in ground_truth:
        paper_name = ground_truth['paper']
        span = ground_truth['chunk_span']
        document_text = load_document(paper_name, md_dir, document_cache)
        return [document_text[span['start']:span['end']]]
    
    else:
        gt_texts = []
        for span in ground_truth['chunk_spans']:
            paper_name = span.get('paper', ground_truth['paper'])
            document_text = load_document(paper_name, md_dir, document_cache)
            if document_text:
                gt_texts.append(document_text[span['start']:span['end']])
        return gt_texts


def chunk_accuracy_metrics(retrieved_chunks, ground_truth, md_dir, document_cache=None):
    """Calculate all chunk accuracy metrics"""
    gt_texts = get_ground_truth_texts(ground_truth, md_dir, document_cache)
    
    if not gt_texts or not retrieved_chunks:
        return {
            't1o': 0,
            'bo': 0,
            'ao': 0
        }
    
    # T1O: Top-1 Overlap
    top1_overlaps = [calculate_text_overlap(retrieved_chunks[0]['text'], gt_text) for gt_text in gt_texts]
    t1o = max(top1_overlaps) if top1_overlaps else 0
    
    # BO: Best Overlap
    bo = 0
    for chunk in retrieved_chunks:
        for gt_text in gt_texts:
            overlap = calculate_text_overlap(chunk['text'], gt_text)
            bo = max(bo, overlap)
    
    # AO: Average Overlap
    all_overlaps = []
    for chunk in retrieved_chunks:
        chunk_max_overlap = max([calculate_text_overlap(chunk['text'], gt_text) for gt_text in gt_texts]) if gt_texts else 0
        all_overlaps.append(chunk_max_overlap)
    ao = sum(all_overlaps) / len(all_overlaps) if all_overlaps else 0
    
    return {
        't1o': t1o,
        'bo': bo,
        'ao': ao
    }


def compute_metrics(retrieved_chunks, ground_truth, md_dir, chunk_scores=None, document_cache=None):
    """Compute all metrics for a single query"""
    gold_metrics = gold_span_metrics(retrieved_chunks, ground_truth, chunk_scores)
    chunk_metrics = chunk_accuracy_metrics(retrieved_chunks, ground_truth, md_dir, document_cache)
    
    return {**gold_metrics, **chunk_metrics}
