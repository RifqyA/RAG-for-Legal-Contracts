"""
RAG Pipeline (deduplicated from notebook)
Auto-generated from RAG_PIPELINE_MAIN.ipynb.
- Imports are consolidated.
- Duplicate function/class definitions removed (keep first occurrence).
- Notebook magics and shell commands are omitted.
"""


import re, json, logging, hashlib, pathlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from functools import lru_cache
import fitz  # PyMuPDF
import langid
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.chat_models import init_chat_model
from mistralai import Mistral
import os
from google.colab import userdata, files
from statistics import median
import logging
import torch
import math
from langchain_huggingface import HuggingFaceEmbeddings
import re, os
from typing import List, Tuple, Dict
import pathlib, hashlib, re
    import numpy as np
            from transformers import AutoTokenizer, AutoModel
import re, json, logging
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
    from transformers import pipeline


# ---- Definitions ----


def get_pdf_via_upload():
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError('No file uploaded.')
    filename = next(iter(uploaded.keys()))
    if not filename.lower().endswith('.pdf'):
        raise ValueError(f'Uploaded file is not a PDF: {filename}')
    return filename


def _build_y_hist(words, H, bins=200, smooth_k=5):
    """
    Build a vertical word-density histogram (y-axis). We bin words by their
    y–center and smooth the counts. This helps spot low-density bands near
    the top/bottom of a page (typical header/footer zones).
    """
    hist = [0.0] * bins
    for x0, y0, x1, y1, *_ in words:
        if x1 <= x0 or y1 <= y0:
            continue
        y0c = max(0.0, min(H, y0))
        y1c = max(0.0, min(H, y1))
        if y1c <= y0c:
            continue
        b0 = max(0, min(bins - 1, int(y0c / H * bins)))
        b1 = max(0, min(bins - 1, int(y1c / H * bins)))
        w = max(0.5, y1c - y0c)
        for b in range(b0, b1 + 1):
            hist[b] += w
    if smooth_k > 1:
        k = int(smooth_k)
        half = k // 2
        sm = []
        for i in range(bins):
            s = 0.0
            c = 0
            for j in range(i - half, i + half + 1):
                if 0 <= j < bins:
                    s += hist[j]
                    c += 1
            sm.append(s / c if c else 0.0)
        hist = sm
    return (hist, sum(hist))


def _detect_header_y(words, H, *, bins=200, smooth_k=5, top_cum=0.01, top_bleed=4.0, max_header_frac=0.14):
    """
    Heuristically locate the header band:
    1) Use the top portion of the histogram (top_cum) to find a low-density
      “valley” just below the header text.
    2) Limit header height by max_header_frac to avoid over-cropping.
    Returns the y-coordinate where body text should start.
    """
    hist, total = _build_y_hist(words, H, bins=bins, smooth_k=smooth_k)
    if total <= 0:
        return H * max_header_frac
    need = total * float(top_cum)
    cum = 0.0
    idx = 0
    for i, v in enumerate(hist):
        cum += v
        if cum >= need:
            idx = i
            break
    y_cut = idx / bins * H - top_bleed
    y_max = H * max_header_frac
    return max(0.0, min(y_cut, y_max))


def _detect_footer_y(words, H, *, bins=200, smooth_k=5, bottom_cum=0.01, bottom_bleed=4.0, max_footer_frac=0.14):
    """
    Heuristically locate the footer band using the bottom portion of the
    histogram and a low-density “valley” above the footer text. Returns the
    y-coordinate where body text should end.
    """
    hist, total = _build_y_hist(words, H, bins=bins, smooth_k=smooth_k)
    if total <= 0:
        return H * (1.0 - max_footer_frac)
    need = total * float(bottom_cum)
    cum = 0.0
    idx = len(hist) - 1
    for i in range(len(hist) - 1, -1, -1):
        cum += hist[i]
        if cum >= need:
            idx = i
            break
    y_cut = idx / bins * H + bottom_bleed
    y_min = H * (1.0 - max_footer_frac)
    return max(y_min, min(H, y_cut))


def _find_gutter_x(words, page_w, *, bins=48, center_window=(0.35, 0.65), min_side_frac=0.25, rel_empty_ratio=0.15, min_center_sep_frac=0.18):
    """
    Two-column detector:
    - Split page width into vertical bands and check that the middle window
      (center_window) is relatively empty across many bands.
    - If so, estimate a “gutter” x that separates left/right columns.
    Returns gutter x or None if the page looks single-column.
    """
    if not words:
        return (False, None)
    xs = [0.5 * (w[0] + w[2]) for w in words]
    hist = [0] * bins
    for x in xs:
        i = min(bins - 1, max(0, int(x / page_w * bins)))
        hist[i] += 1
    lo = int(center_window[0] * bins)
    hi = int(center_window[1] * bins)
    if hi <= lo:
        return (False, None)
    mid = min(range(lo, hi), key=lambda i: hist[i])
    left_count = sum(hist[:mid])
    right_count = sum(hist[mid + 1:])
    total = left_count + right_count + hist[mid]
    if total == 0:
        return (False, None)
    if left_count / total < min_side_frac or right_count / total < min_side_frac:
        return (False, None)
    side_bins = mid + (bins - 1 - mid) or 1
    side_mean = (left_count + right_count) / side_bins
    if hist[mid] > rel_empty_ratio * max(1, side_mean):
        return (False, None)
    left_xs = [x for x in xs if x / page_w <= mid / bins]
    right_xs = [x for x in xs if x / page_w > mid / bins]
    if not left_xs or not right_xs:
        return (False, None)
    sep = (sum(right_xs) / len(right_xs) - sum(left_xs) / len(left_xs)) / page_w
    if sep < min_center_sep_frac:
        return (False, None)
    return (True, (mid + 0.5) * (page_w / bins))


def _group_into_lines(words, y_tol):
    """
    Group word boxes into line objects (x0, x1, y_center, text, height).
    This helps us decide which words belong to the same printed line.
    """
    words = sorted(words, key=lambda w: (w[1], w[0]))
    lines, cur, cur_y = ([], [], None)
    for w in words:
        yc = 0.5 * (w[1] + w[3])
        if cur_y is None or abs(yc - cur_y) <= y_tol:
            cur.append(w)
            cur_y = yc if cur_y is None else 0.7 * cur_y + 0.3 * yc
        else:
            cur.sort(key=lambda t: t[0])
            lines.append(' '.join((tok[4] for tok in cur)))
            cur = [w]
            cur_y = yc
    if cur:
        cur.sort(key=lambda t: t[0])
        lines.append(' '.join((tok[4] for tok in cur)))
    return lines


def _pick_english_column(left_text: str, right_text: str, *, langid_thr: float=0.94, min_chars: int=40):
    """
    Given left/right column text, use langid to choose the English column
    with minimum length and confidence thresholds. Falls back safely if
    both are non-English or too short.
    """

    def analyze(t: str):
        t = ' '.join((t or '').split())
        if not t:
            return {'lang': '', 'conf': 0.0, 'en_rank': float('-inf'), 'len': 0}
        lang, conf = langid.classify(t)
        en_rank = float('-inf')
        try:
            for lg, sc in langid.rank(t):
                if lg == 'en':
                    en_rank = sc
                    break
        except Exception:
            pass
        return {'lang': lang, 'conf': conf, 'en_rank': en_rank, 'len': len(t)}
    L = analyze(left_text)
    R = analyze(right_text)
    if L['lang'] == 'en' and L['conf'] >= langid_thr and (not (R['lang'] == 'en' and R['conf'] >= langid_thr)):
        return ('left', L, R)
    if R['lang'] == 'en' and R['conf'] >= langid_thr and (not (L['lang'] == 'en' and L['conf'] >= langid_thr)):
        return ('right', L, R)
    if L['lang'] == 'en' and L['conf'] >= langid_thr and (R['lang'] == 'en') and (R['conf'] >= langid_thr):
        return ('left' if L['conf'] >= R['conf'] else 'right', L, R)
    if L['en_rank'] != R['en_rank']:
        return ('left' if L['en_rank'] > R['en_rank'] else 'right', L, R)
    if L['len'] != R['len']:
        return ('left' if L['len'] >= R['len'] else 'right', L, R)
    return ('left', L, R)


def outputcom(pdf_path, *, debug=False, print_pages=False, max_pages=None, return_stats=False, min_body_frac=0.55):
    """
    PDF preprocessor Workflow:
    1) Opens the PDF with PyMuPDF (fitz) and extracts word boxes per page.
    2) Detects header/footer bands via y-histogram valleys; crops them out.
    3) Detects two-column layout via center “gutter” emptiness.
    4) If two columns, selects the English column using langid.
    5) Concatenates cleaned page text (optionally returns stats for debug).
    """
    logger = logging.getLogger('outputcom')
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(h)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        logger.exception('Failed to open PDF: %s', pdf_path)
        raise
    out_text, stats = ([], [])
    for pno, page in enumerate(doc, start=1):
        if max_pages and pno > max_pages:
            break
        try:
            W, H = (page.rect.width, page.rect.height)
            words = page.get_text('words')
            if not words:
                logger.info('Page %d: no text', pno)
                out_text.append('')
                stats.append({'page': pno, 'total_words': 0})
                continue
            y_head_end = _detect_header_y(words, H)
            y_foot_end = _detect_footer_y(words, H)
            min_body_h = H * float(min_body_frac)
            if y_foot_end - y_head_end < min_body_h:
                mid = 0.5 * (y_head_end + y_foot_end)
                y_head_end = max(0.0, mid - 0.5 * min_body_h)
                y_foot_end = min(H, mid + 0.5 * min_body_h)
            body_words = [w for w in words if y_head_end <= 0.5 * (w[1] + w[3]) < y_foot_end]
            if not body_words:
                logger.info('Page %d: empty after header/footer removal', pno)
                out_text.append('')
                stats.append({'page': pno, 'total_words': len(words), 'kept_words': 0})
                continue
            h_med = median([max(1.0, w[3] - w[1]) for w in body_words])
            y_tol = 0.6 * h_med
            is_two, gutter_x = _find_gutter_x(body_words, W)
            if is_two and gutter_x is not None:
                left = [w for w in body_words if 0.5 * (w[0] + w[2]) <= gutter_x]
                right = [w for w in body_words if 0.5 * (w[0] + w[2]) > gutter_x]
                if not left or not right:
                    lines = _group_into_lines(body_words, y_tol)
                    page_text = '\n'.join(lines)
                    chosen = 'single'
                    Linfo = {'lang': '', 'conf': 0.0}
                    Rinfo = {'lang': '', 'conf': 0.0}
                else:
                    left_lines = _group_into_lines(left, y_tol)
                    right_lines = _group_into_lines(right, y_tol)
                    left_text = '\n'.join(left_lines)
                    right_text = '\n'.join(right_lines)
                    chosen, Linfo, Rinfo = _pick_english_column(left_text, right_text)
                    page_text = left_text if chosen == 'left' else right_text
            else:
                lines = _group_into_lines(body_words, y_tol)
                page_text = '\n'.join(lines)
                chosen = 'single'
                Linfo = {'lang': '', 'conf': 0.0}
                Rinfo = {'lang': '', 'conf': 0.0}
            if print_pages:
                hdr = f'\n=== Page {pno} (chosen={chosen}; head={y_head_end:.1f}; foot={y_foot_end:.1f}) ===\n'
                print(hdr, end='')
                print(page_text)
            out_text.append(page_text)
            stats.append({'page': pno, 'two_cols': bool(is_two and gutter_x is not None), 'chosen': chosen, 'left_lang': Linfo.get('lang'), 'left_conf': float(f"{Linfo.get('conf', 0.0):.3f}"), 'right_lang': Rinfo.get('lang'), 'right_conf': float(f"{Rinfo.get('conf', 0.0):.3f}"), 'header_y': float(f'{y_head_end:.1f}'), 'footer_y': float(f'{y_foot_end:.1f}')})
        except Exception:
            logger.exception('Page %d: processing error', pno)
            out_text.append('')
            stats.append({'page': pno, 'error': 'processing failure'})
    doc.close()
    combined = '\n\n'.join(out_text)
    return (combined, stats) if return_stats else combined


def is_header_line(line: str) -> bool:
    """
    Return True if a line looks like a section heading of the form '4. Title'
    (single leading number + dot + space). No multi-level numbers allowed.
    NOTE: Bold enforcement is applied in split_text_by_headings().
    """
    return bool(_SINGLE_NUM_DOT.match((line or '').strip()))


def _norm(s: str) -> str:
    return re.sub('\\s+', ' ', (s or '').strip()).lower()


def _collect_bold_lines(pdf_path: str) -> set:
    """
    Extract lines that contain any bold span (font name includes 'bold').
    Returns a set of normalized bold line strings.
    """
    if pdf_path in _BOLD_LINES_CACHE:
        return _BOLD_LINES_CACHE[pdf_path]
    bold = set()
    doc = fitz.open(pdf_path)
    try:
        for pg in doc:
            data = pg.get_text('dict')
            for block in data.get('blocks', []):
                if block.get('type', 0) != 0:
                    continue
                for line in block.get('lines', []):
                    spans = line.get('spans', [])
                    if any(('bold' in sp.get('font', '').lower() for sp in spans)):
                        text = ''.join((sp.get('text', '') for sp in spans)).strip()
                        if len(text) >= 3:
                            bold.add(_norm(text))
    finally:
        doc.close()
    _BOLD_LINES_CACHE[pdf_path] = bold
    return bold


def split_text_by_headings(text: str, *, source_path: str) -> List[Tuple[str, str]]:
    """
    Split the preprocessed text into (heading, content) pairs using:
      - Strict 'N. Title' pattern (single number + '.' + space)
      - ALWAYS-ON bold enforcement: the heading line must appear bold in the PDF
    Preserves 'Preamble' for any text before the first accepted heading.
    """
    lines = text.splitlines()
    chunks: List[Tuple[str, str]] = []
    current_heading = None
    buffer: List[str] = []
    bold_lines = _collect_bold_lines(source_path)

    def is_bold_heading(h: str) -> bool:
        if not is_header_line(h):
            return False
        nh = _norm(h)
        if nh in bold_lines:
            return True
        return any((nh.startswith(b) or nh.endswith(b) or b.startswith(nh) or b.endswith(nh) for b in bold_lines if len(b) >= 6 and len(nh) >= 6))
    for line in lines:
        if is_bold_heading(line):
            if current_heading is not None:
                chunks.append((current_heading, '\n'.join(buffer).strip()))
            else:
                pre = '\n'.join(buffer).strip()
                if pre:
                    chunks.append(('Preamble', pre))
            current_heading = line.strip()
            buffer = []
        else:
            buffer.append(line)
    if current_heading is not None:
        chunks.append((current_heading, '\n'.join(buffer).strip()))
    else:
        chunks.append(('Preamble', '\n'.join(buffer).strip()))
    return [(h, c) for h, c in chunks if c and c.strip()]


def _heading_nums(h: str) -> List[int]:
    return [int(x) for x in re.findall('\\d+', h)]


def build_documents_from_text(text: str, source: str) -> List['Document']:
    """
    Creates Documents with:
      - heading
      - section_id (monotonic order)
      - heading_level (len of numbering)
      - h_prefix_i for exact-match filtering (e.g., '2', '2.1', '2.1.3')
      - source (doc id/path)
    Also populates SECTION_INDEX and DOC_ORDER for neighbor expansion.
    """
    pairs = split_text_by_headings(text, source_path=source)
    docs: List['Document'] = []
    for section_id, (heading, content) in enumerate(pairs):
        nums = _heading_nums(heading)
        level = len(nums)
        prefixes = ['.'.join((str(n) for n in nums[:i])) for i in range(1, level + 1)]
        meta = {'source': source, 'heading': heading.strip(), 'heading_level': level, 'section_id': section_id}
        for i, p in enumerate(prefixes, start=1):
            meta[f'h_prefix_{i}'] = p
        doc = Document(page_content=content, metadata=meta)
        docs.append(doc)
        SECTION_INDEX[source, section_id] = doc
        DOC_ORDER[source].append(section_id)
    return docs


def preview_splits(text: str, max_preview_chars: int=280, show_empty: bool=True) -> None:
    """
    #Prints a summary table and short previews for each (heading, content) chunk.
"""
    chunks = split_text_by_headings(text, source_path=pdf_path)
    rows = []
    for i, (heading, content) in enumerate(chunks, start=1):
        chars = len(content)
        words = len(content.split())
        if not show_empty and chars == 0:
            continue
        rows.append((i, heading, chars, words))
    print(f'\nTotal chunks: {len(chunks)}')
    print(f"{'Idx':>3}  {'Heading':<60} {'Chars':>7} {'Words':>7}")
    print('-' * 84)
    for i, heading, chars, words in rows:
        h = heading[:57] + '…' if len(heading) > 60 else heading
        print(f'{i:>3}  {h:<60} {chars:>7} {words:>7}')
    print('\n=== Detailed previews ===')
    for i, (heading, content) in enumerate(chunks, start=1):
        preview = ' '.join(content.split())
        if len(preview) > max_preview_chars:
            preview = preview[:max_preview_chars] + '…'
        print(f'\n[{i}] {heading}')
        print(preview if preview else '<empty>')


def slugify(s: str) -> str:
    return re.sub('[^a-z0-9]+', '_', s.lower()).strip('_')


def make_source_id(pdf_path: str, *, version: str | None=None, content_text: str | None=None) -> str:
    """
    Build a stable doc_id for upserts & provenance:
      <slug_of_filename>[__v-<version>][__h-<hash12>]
    Tip: put content hash in metadata too, not only in the ID, to avoid over-splitting IDs.
    """
    name = pathlib.Path(pdf_path).stem
    base = slugify(name)
    parts = [base]
    if version:
        parts.append(f'v_{slugify(version)}')
    if content_text:
        h = hashlib.sha1(content_text.encode('utf-8')).hexdigest()[:12]
        parts.append(f'h_{h}')
    return '__'.join(parts)


def build_documents_from_text(text: str, *, pdf_path: str, doc_id: str) -> list['Document']:
    """
    - Uses pdf_path for bold detection (PyMuPDF).
    - Uses doc_id for stable indexing/upserts and neighbor keys.
    """
    pairs = split_text_by_headings(text, source_path=pdf_path)
    docs: list['Document'] = []
    for section_id, (heading, content) in enumerate(pairs):
        nums = [int(x) for x in re.findall('\\d+', heading)]
        level = len(nums)
        prefixes = ['.'.join((str(n) for n in nums[:i])) for i in range(1, level + 1)]
        meta = {'doc_id': doc_id, 'source_path': pdf_path, 'heading': heading.strip(), 'heading_level': level, 'section_id': section_id, 'content_hash': hashlib.sha1(content.encode('utf-8')).hexdigest()[:12]}
        for i, p in enumerate(prefixes, start=1):
            meta[f'h_prefix_{i}'] = p
        doc = Document(page_content=content, metadata=meta)
        docs.append(doc)
        SECTION_INDEX[doc_id, section_id] = doc
        DOC_ORDER[doc_id].append(section_id)
    return docs


def build_bm25_retriever(all_docs: List[Document], k: int=50) -> BM25Retriever:
    """
    Build a local BM25 retriever (sparse lexical matching).
    - Uses rank_bm25 under the hood via LangChain's BM25Retriever.
    - Optional normalization/tokenization for better matching.
    We’ll fuse this with vector results to reduce embedding blind spots.
    """
    if not all_docs:
        raise ValueError('all_docs is empty')
    fdocs = [d for d in all_docs if (d.page_content or '').strip()]
    if not fdocs:
        raise ValueError('No non-empty documents')
    r = BM25Retriever.from_documents(fdocs)
    r.k = k

    def _tok(s: str):
        return re.findall('[A-Za-z0-9]+', (s or '').lower())
    if hasattr(r, 'preprocess_func'):
        r.preprocess_func = _tok
    return r


def _rx_score(text: str, patterns) -> float:
    """Count how many patterns are present (compiled or raw)."""
    return float(sum((1 for p in patterns if (p.search(text) if hasattr(p, 'search') else re.search(p, text, flags=re.I)))))


def _fuzzy_seed_score(query: str, label: str) -> float:
    seed = _CLASS_SEEDS.get(label, '')
    return fuzz.token_set_ratio(query, seed) / 100.0


def classify_intent_with_conf(q: str):
    """
    Lightweight intent classifier with fuzzy priors:
    Labels: RIGHTS, OBLIGATIONS, PROHIBITIONS, RISK.
    We bias retrieval/ranking based on the user's goal to pull the most
    relevant sections (e.g., caps/limitations when intent ≈ RISK).
    """
    q = q or ''
    scores = {'RIGHTS': _rx_score(q, RIGHT_SIGNS_RX), 'OBLIGATIONS': _rx_score(q, OBLIG_SIGNS_RX), 'PROHIBITIONS': _rx_score(q, PROH_SIGNS_RX), 'RISK': _rx_score(q, RISK_SIGNS_RX)}
    for lbl in scores:
        scores[lbl] += 0.5 * _fuzzy_seed_score(q, lbl)
    total = sum(scores.values())
    if total <= 1e-09:
        return ('GENERAL', 0.0, scores)
    probs = {k: v / total for k, v in scores.items()}
    label = max(probs, key=probs.get)
    conf = probs[label]
    if conf < 0.45:
        return ('GENERAL', conf, probs)
    return (label, conf, probs)


def classify_intent(q: str) -> str:
    label, _conf, _ = classify_intent_with_conf(q)
    return label


def intent_bias(doc_text: str, heading: str, intent: str) -> float:
    """Tiny additive bias by intent (same math as before)."""
    t = f'{heading}\n{doc_text}'
    if intent == 'RIGHTS':
        return +1.0 * _rx_score(t, RIGHT_SIGNS_RX) - 0.8 * _rx_score(t, OBLIG_SIGNS_RX) - 0.4 * _rx_score(t, PROH_SIGNS_RX) - 0.2 * _rx_score(t, RISK_SIGNS_RX)
    if intent == 'OBLIGATIONS':
        return +1.0 * _rx_score(t, OBLIG_SIGNS_RX) - 0.8 * _rx_score(t, RIGHT_SIGNS_RX) - 0.2 * _rx_score(t, PROH_SIGNS_RX)
    if intent == 'PROHIBITIONS':
        return +1.0 * _rx_score(t, PROH_SIGNS_RX) - 0.6 * _rx_score(t, RIGHT_SIGNS_RX) - 0.2 * _rx_score(t, OBLIG_SIGNS_RX)
    if intent == 'RISK':
        return +1.0 * _rx_score(t, RISK_SIGNS_RX) - 0.3 * _rx_score(t, RIGHT_SIGNS_RX) - 0.3 * _rx_score(t, OBLIG_SIGNS_RX)
    return 0.0


@lru_cache(maxsize=16384)
def _cached_token_set_ratio(q: str, h: str) -> float:
    return fuzz.token_set_ratio(q, h) / 100.0


def heading_fuzzy_boost(query: str, heading: str) -> float:
    """Tiny semantic nudge if heading wording aligns with the query."""
    return _cached_token_set_ratio(query or '', heading or '')


def _looks_like_definitions(d) -> bool:
    h = (getattr(d, 'metadata', {}).get('heading') or '').upper()
    return 'DEFINITIONS' in h or 'INTERPRETATION' in h or 'PREAMBLE' in h


def _format_examples(examples: List[Tuple[str, Dict[str, float]]]) -> str:
    lines = []
    for clause, scores in examples:
        s = {k: float(scores.get(k, 0.0)) for k in LABELS}
        lines.append(f'Clause: {clause}\nScores: {json.dumps(s)}')
    return 'Examples (labeled):\n' + '\n\n'.join(lines) + '\n'


def ml_scores_one_shot(text: str, *, examples: Optional[List[Tuple[str, Dict[str, float]]]]=None, model: str='mistral-large-latest', temperature: float=0.0) -> Dict[str, float]:
    """
    Offline scorer using Legal-BERT embeddings.
    Returns a JSON-like dict with keys: RIGHTS, OBLIGATIONS, PROHIBITIONS, RISK (values in 0..1).
    If LEGALBERT_PROTOS (centroids) is available (built from your corpus), uses it;
    otherwise falls back to embedding short prototype texts.

    Args kept for compatibility:
      - examples: ignored
      - model: ignored
      - temperature: if >0, apply softmax with 1/temperature; else L1-normalize positive sims.
    """
    import numpy as np
    try:
        _encode_mean
    except NameError:

        def _lazy_load_legalbert():
            import torch
            from transformers import AutoTokenizer, AutoModel
            name = 'nlpaueb/legal-bert-base-uncased'
            tok = AutoTokenizer.from_pretrained(name)
            enc = AutoModel.from_pretrained(name)
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            enc.to(dev)
            return (tok, enc, dev)

        def _encode_mean(texts: List[str]):
            import torch
            tok, enc, dev = _lazy_load_legalbert()
            outs = []
            with torch.no_grad():
                for i in range(0, len(texts), 8):
                    batch = texts[i:i + 8]
                    X = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(dev)
                    Y = enc(**X).last_hidden_state
                    mask = X['attention_mask'].unsqueeze(-1)
                    mean = (Y * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    outs.append(mean.cpu().numpy())
            return np.vstack(outs)
    try:
        LABELS
    except NameError:
        LABELS = ['RIGHTS', 'OBLIGATIONS', 'PROHIBITIONS', 'RISK']
    fallback_proto_text = {'RIGHTS': 'has the right to; may; entitled; license granted.', 'OBLIGATIONS': 'shall; must; required to; responsibilities; duties.', 'PROHIBITIONS': 'shall not; may not; prohibited; restriction.', 'RISK': 'liability; indemnification; limitation of liability; damages cap.'}

    def _intent_centroids():
        """
        Return (labels, centroids[4,H]) either from LEGALBERT_PROTOS or fallback strings.
        Centroids are L2-normalized for cosine.
        """
        C = None
        try:
            if 'LEGALBERT_PROTOS' in globals() and LEGALBERT_PROTOS and ('centroids' in LEGALBERT_PROTOS):
                labs = LEGALBERT_PROTOS['keys']
                C = LEGALBERT_PROTOS['centroids']
                C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-09)
                return (labs, C)
        except Exception:
            pass
        labs = LABELS
        P = _encode_mean([fallback_proto_text[k] for k in labs])
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-09)
        return (labs, P)

    def _cosine_to_centroids(vec: np.ndarray, C: np.ndarray) -> np.ndarray:
        v = vec / (np.linalg.norm(vec) + 1e-09)
        return (C @ v.reshape(-1, 1)).ravel()
    text = text or ''
    labs, C = _intent_centroids()
    v = _encode_mean([text])[0]
    sims = _cosine_to_centroids(v, C)
    sims = np.clip(sims, 0.0, None)
    if temperature and temperature > 0:
        tau = max(1e-06, float(temperature))
        x = sims / tau
        x = x - x.max()
        exp = np.exp(x)
        probs = exp / (exp.sum() + 1e-09)
    else:
        s = sims.sum()
        probs = sims / (s + 1e-09)
    out = {lab: float(prob) for lab, prob in zip(labs, probs)}
    return {k: float(out.get(k, 0.0)) for k in LABELS}


@dataclass
class RetrievalPlan:
    subqueries: List[Tuple[str, float]]
    need_definitions: bool
    notes: str = ''


def _lazy_load_legalbert():
    """Load tokenizer+models once; CPU by default, CUDA if available."""
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
    key = 'legalbert'
    if key in _LEGALBERT_CACHE:
        return _LEGALBERT_CACHE[key]
    name = 'nlpaueb/legal-bert-base-uncased'
    tok = AutoTokenizer.from_pretrained(name)
    mlm = AutoModelForMaskedLM.from_pretrained(name)
    enc = AutoModel.from_pretrained(name)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlm.to(dev)
    enc.to(dev)
    _LEGALBERT_CACHE[key] = (tok, mlm, enc, dev)
    return (tok, mlm, enc, dev)


def _fill_mask_candidates(query: str, top_k: int=6, per_tpl: int=6) -> List[str]:
    """
    Legal-BERT masked-LM expansions to catch domain terms (e.g., 'indemnification',
    'force majeure'). Produces short candidate tokens/phrases.
    """
    from transformers import pipeline
    tok, mlm, _, _ = _lazy_load_legalbert()
    mask = tok.mask_token
    templates = [f'In a contract, a clause about {query} is called {mask}.', f'Legal term related to {query}: {mask}.', f'Another term for {query} is {mask}.', f'Common issue around {query} is {mask}.']
    ngrams = set()
    fill = pipeline('fill-mask', model=mlm, tokenizer=tok, top_k=per_tpl, device=-1)
    for t in templates:
        try:
            outs = fill(t)
            for o in outs:
                token = (o.get('token_str') or '').strip().lower()
                if 3 <= len(token) <= 24 and re.fullmatch('[a-z][a-z\\-]*', token):
                    ngrams.add(token)
        except Exception as e:
            logger.debug('MLM expansion skipped: %s', e)
    return sorted(ngrams)


def _encode_mean(texts: List[str]):
    """Mean-pool last hidden state as a simple sentence embedding."""
    import torch
    import numpy as np
    tok, _, enc, dev = _lazy_load_legalbert()
    with torch.no_grad():
        outs = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i + 8]
            X = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(dev)
            Y = enc(**X).last_hidden_state
            mask = X['attention_mask'].unsqueeze(-1)
            mean = (Y * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outs.append(mean.cpu().numpy())
        return np.vstack(outs)


def _intent_cues(query: str) -> Dict[str, float]:
    """
    Lightweight, local intent scores (RIGHTS/OBLIGATIONS/PROHIBITIONS/RISK)
    via cosine similarity to tiny prototype prompts — no API needed.
    """
    import numpy as np
    protos = {'RIGHTS': 'has the right to; may; entitled; license granted.', 'OBLIGATIONS': 'shall; must; required to; responsibilities; duties.', 'PROHIBITIONS': 'shall not; may not; prohibited; restriction.', 'RISK': 'liability; indemnification; limitation of liability; damages cap.'}
    texts = [query] + list(protos.values())
    E = _encode_mean(texts)
    q, P = (E[0:1], E[1:])
    sims = q @ P.T / (np.linalg.norm(q) * np.linalg.norm(P, axis=1) + 1e-09)
    sims = sims.clip(0, None)
    sims = sims / (sims.sum() + 1e-09)
    return {k: float(v) for k, v in zip(protos.keys(), sims)}


def reason_to_retrieve(query: str, llm=None, max_subqueries: int=7) -> RetrievalPlan:
    """
    Builds a retrieval plan using:
      1) Section refs + defined-term detection (your original logic),
      2) Hand-rolled synonyms (small but precise),
      3) **Legal-BERT MLM expansions** (domain terms, offline),
      4) Tiny **Legal-BERT intent cues** to bias later ranking,
      5) (Optional) LLM assist remains supported but is no longer required.
    """
    q = (query or '').strip()
    subs: List[Tuple[str, float]] = []
    need_defs = False
    notes = []
    sec_refs = re.findall('(?:§|\\bsection\\s+)(\\d+(?:\\.\\d+)*)', q, flags=re.I)
    for sec in sec_refs:
        subs.append((f'Section {sec}', 0.8))
    if sec_refs:
        notes.append(f'sections={sec_refs}')
    caps = re.findall('\\b([A-Z][A-Za-z]+(?:\\s+[A-Z][A-Za-z]+){0,3})\\b', q)
    defs_terms = [t for t in caps if t not in _CAP_STOP and len(t.split()) >= 2]
    if defs_terms:
        need_defs = True
        for t in defs_terms[:3]:
            subs.append((f'Definition of {t}', 0.7))
        notes.append(f'defs={defs_terms}')
    for pat, alts in _SYNONYMS.items():
        if re.search(pat, q, flags=re.I):
            for a in alts[:3]:
                subs.append((re.sub(pat, a, q, flags=re.I), 0.62))
            notes.append(f'syn[{pat}]→{alts[:3]}')
    try:
        mlm_terms = _fill_mask_candidates(q, top_k=6, per_tpl=6)
        for t in mlm_terms[:8]:
            subs.append((f'{q} {t}', 0.68))
        if mlm_terms:
            notes.append(f'mlm+={mlm_terms[:8]}')
    except Exception as e:
        logger.debug('Legal-BERT MLM off: %s', e)
    if llm:
        try:
            prompt = f'Give 3–5 alternative searches for this legal question, no bullets.\nQ: {q}\nOnly short queries, one per line.'
            raw = llm.invoke(prompt).content if hasattr(llm, 'invoke') else llm.predict(prompt)
            for line in (raw or '').splitlines():
                s = line.strip(' -•\t')
                if 6 <= len(s) <= 140:
                    subs.append((s, 0.65))
            notes.append('llm+')
        except Exception as e:
            logger.debug('LLM assist skipped: %s', e)
    try:
        ic = _intent_cues(q)
        notes.append(f'intent={json.dumps(ic)}')
        globals()['LEGALBERT_INTENT'] = ic
    except Exception as e:
        logger.debug('Intent cues skipped: %s', e)
    uniq, seen = ([], set())
    for text, w in subs:
        key = (text or '').strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append((text.strip(), float(max(0.1, min(1.0, w)))))
    plan = RetrievalPlan(subqueries=uniq[:max_subqueries], need_definitions=need_defs, notes='; '.join(notes))
    logger.info('RT-plan: %s | subs=%s', plan.notes, [s for s, _ in plan.subqueries])
    return plan


def _squash_ws(s: str) -> str:
    return re.sub('\\s+', ' ', s or '').strip()


def _serialize_preview(docs: List[Document], *, max_chars_per_doc: int=300, max_total_chars: int=2500) -> str:
    """Build a compact preview for printing only."""
    blocks = []
    used = 0
    for i, d in enumerate(docs, start=1):
        heading = d.metadata.get('heading') or d.metadata.get('title') or '<no heading>'
        src = d.metadata.get('source', '')
        sec = d.metadata.get('section_id', '')
        snippet = _squash_ws(d.page_content)[:max_chars_per_doc]
        block = f'[#{i}] {heading} (source={src}, section={sec})\n{snippet}\n'
        if used + len(block) > max_total_chars:
            break
        blocks.append(block)
        used += len(block)
    return '\n---\n'.join(blocks)


def _embedder_name(e) -> str:
    for attr in ('model', 'model_name', 'model_name_or_path', 'deployment', '__class__'):
        if hasattr(e, attr):
            val = getattr(e, attr)
            return val if isinstance(val, str) else getattr(val, '__name__', str(val))
    return str(e)


def _vec_stats(vec) -> Dict[str, Any]:
    try:
        v = list(map(float, vec))
    except Exception:
        return {'dim': None, 'norm': None, 'min': None, 'max': None, 'head5': None, 'sha1_8': None}
    dim = len(v)
    norm = math.sqrt(sum((x * x for x in v))) if dim else 0.0
    vmin = min(v) if dim else None
    vmax = max(v) if dim else None
    head5 = v[:5]
    hsrc = ','.join((f'{x:.6f}' for x in head5)) + f'|{dim}|{norm:.6f}'
    sha1_8 = hashlib.sha1(hsrc.encode('utf-8')).hexdigest()[:8]
    return {'dim': dim, 'norm': norm, 'min': vmin, 'max': vmax, 'head5': head5, 'sha1_8': sha1_8}


def serialize_preview(docs: List[Document], *, max_chars_per_doc: int=300, max_total_chars: int=2500) -> str:
    """
    Build a compact, human-friendly preview string from docs.
    Enforces both per-doc and overall character caps.
    """
    lines: List[str] = []
    used = 0
    for doc in docs:
        src_line = f'Source: {doc.metadata}\nContent: '
        src_len = len(src_line)
        if used + src_len >= max_total_chars:
            break
        content = doc.page_content or ''
        if len(content) > max_chars_per_doc:
            content = content[:max(0, max_chars_per_doc - 3)] + '...'
        remaining = max_total_chars - used - src_len
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:max(0, remaining - 3)] + '...'
        block = src_line + content
        lines.append(block)
        used += len(block) + 2
        if used >= max_total_chars:
            break
    return '\n\n'.join(lines)


def rerank_docs(question: str, contexts: list[str], batch_size: int=32) -> list[int]:
    """
    Rank contexts by cross-encoder score (higher = better) and
    return their indices in descending score order.
    """
    if not contexts:
        return []
    pairs = [(question, ctx) for ctx in contexts]
    with torch.no_grad():
        scores = CE.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranked_ids = sorted(range(len(contexts)), key=lambda i: float(scores[i]), reverse=True)
    return ranked_ids


def retrieve(query: str, k: int=6, *, window_size: int=1, mmr_lambda: float=0.25, fetch_k: Optional[int]=None, bm25_retriever: Any, bm25_weight: float=0.4, cross_encoder_model: str='cross-encoder/ms-marco-MiniLM-L-6-v2', score_threshold: float=0.0, filters: Optional[dict]=None, query_prefix: Optional[str]=None, embedder: Any=None, preview: bool=True, preview_max_chars_per_doc: int=300, preview_max_total_chars: int=2500) -> Tuple[str, List[Document]]:
    """
    Retrieval pipeline (returns a compact preview string + full docs):
    1) Embed the main query (optionally prefixed, e.g., 'query: ...').
    2) Generate weighted sub-queries (query expansion) and embed each.
    3) Dense search (MMR): for main + sub-queries with λ=mmr_lambda to
      reduce redundancy while maximizing coverage.
    4) Sparse search (BM25): gather lexical candidates for exact terms,
      numbers, and “shall/shall not” language often missed by embeddings.
    5) Fuse candidates: weighted sum of (dense * subquery_weight) and
      (sparse * bm25_weight) → rank, then dedupe.
    6) (Optional) Cross-encoder rerank: re-score top-N with a bi-encoder
      classifier for better order.
    7) Neighbor expansion: for each hit, pull ±window_size sections using
      SECTION_INDEX/DOC_ORDER to keep context contiguous.
    8) Build a **preview** context string under global and per-doc char
      budgets; return (preview_string, docs_list) for generation.

    Knobs you can tune:
    - fetch_k: candidate pool size before fusion/rerank
    - bm25_weight: 0.0..1.0 fusion weight for lexical signals
    - window_size: neighbor expansion radius (±N sections)
    """
    fetch_k = fetch_k or max(k * 5, 10)
    logger.info('retrieve(): k=%d, window=%d, bm25_weight=%.2f, thr=%.3f | ALWAYS: MMR, BM25, rerank (%s)', k, window_size, bm25_weight, score_threshold, cross_encoder_model)
    plan = reason_to_retrieve(query, llm=globals().get('RT_LLM'), max_subqueries=5)
    q_text = f'{query_prefix} {query}'.strip() if query_prefix else query
    _emb = embedder or getattr(vector_store, 'embedding', None)
    if not _emb:
        raise ValueError('No embedding model found: pass `embedder` or set `vector_store.embedding`.')
    if hasattr(_emb, 'embed_query'):
        q_vec = _emb.embed_query(q_text)

        def _embed_one(t: str):
            return _emb.embed_query(t)
    else:
        q_vec = _emb.embed_documents([q_text])[0]

        def _embed_one(t: str):
            return _emb.embed_documents([t])[0]
    try:
        stats = _vec_stats(q_vec)
        logger.info('QUERY EMBEDDED with %s | dim=%s norm=%.6f head5=%s', _embedder_name(_emb), stats['dim'], stats['norm'] or 0.0, stats['head5'])
    except Exception:
        pass
    filter_kwargs = {'filter': filters} if filters else {}
    logger.info('Vector (MMR): fetch_k=%d, lambda=%.2f | subqueries=%d', fetch_k, mmr_lambda, len(plan.subqueries))
    vec_scored: List[tuple[Document, float]] = []
    main_vec_docs = vector_store.max_marginal_relevance_search_by_vector(q_vec, k=fetch_k, fetch_k=fetch_k * 2, lambda_mult=mmr_lambda, **filter_kwargs)
    vec_scored.extend(((d, 1.0) for d in main_vec_docs))
    for subq, w in plan.subqueries:
        s_text = f'{query_prefix} {subq}'.strip() if query_prefix else subq
        s_vec = _embed_one(s_text)
        s_docs = vector_store.max_marginal_relevance_search_by_vector(s_vec, k=fetch_k, fetch_k=fetch_k * 2, lambda_mult=mmr_lambda, **filter_kwargs)
        vec_scored.extend(((d, float(w)) for d in s_docs))
    logger.info('Vector candidates: main=%d, subs=%d → total=%d', len(main_vec_docs), len(vec_scored) - len(main_vec_docs), len(vec_scored))
    if bm25_retriever is None:
        raise ValueError('bm25_retriever must be provided (this version always uses BM25).')
    prev_k = getattr(bm25_retriever, 'k', None)
    req_k = fetch_k
    if prev_k is not None:
        bm25_retriever.k = req_k
    bm25_scored: List[tuple[Document, float]] = []
    bm25_main = bm25_retriever.invoke(query) if hasattr(bm25_retriever, 'invoke') else bm25_retriever.get_relevant_documents(query)
    bm25_scored.extend(((d, 1.0) for d in bm25_main[:req_k]))
    for subq, w in plan.subqueries:
        bm25_sub = bm25_retriever.invoke(subq) if hasattr(bm25_retriever, 'invoke') else bm25_retriever.get_relevant_documents(subq)
        bm25_scored.extend(((d, float(w)) for d in bm25_sub[:req_k]))
    if prev_k is not None:
        bm25_retriever.k = prev_k
    logger.info('BM25 candidates: main=%d, subs=%d → total=%d', len(bm25_main[:req_k]), len(bm25_scored) - len(bm25_main[:req_k]), len(bm25_scored))

    def _key(d: Document) -> str:
        return f"{d.metadata.get('source', '')}|{d.metadata.get('section_id', '')}|{hash(d.page_content)}"
    fused: Dict[str, Dict[str, Any]] = {}
    for d, s in vec_scored:
        fused.setdefault(_key(d), {'doc': d, 'v': 0.0, 'b': 0.0})
        fused[_key(d)]['v'] = max(fused[_key(d)]['v'], float(s))
    for d, s in bm25_scored:
        fused.setdefault(_key(d), {'doc': d, 'v': 0.0, 'b': 0.0})
        fused[_key(d)]['b'] = max(fused[_key(d)]['b'], float(s))
    for item in fused.values():
        item['score'] = (1 - bm25_weight) * item['v'] + bm25_weight * item['b']
    candidates: List[tuple[Document, float]] = [(it['doc'], it['score']) for it in fused.values() if it['score'] >= score_threshold]
    candidates.sort(key=lambda x: x[1], reverse=True)
    logger.info('Fusion: vec=%d, bm25=%d → unique=%d (thr=%.3f)', len(vec_scored), len(bm25_scored), len(candidates), score_threshold)
    logger.info('Cross-encoder rerank: model=%s, n=%d', CROSS_ENCODER_MODEL, len(candidates))
    pairs = [(query, d.page_content) for d, _ in candidates]
    with torch.no_grad():
        ce_scores = CE.predict(pairs, batch_size=32, show_progress_bar=False)
    candidates = [(d, float(s)) for (d, _), s in zip(candidates, ce_scores)]
    candidates.sort(key=lambda x: x[1], reverse=True)
    logger.info('Rerank completed')
    intent_label, intent_conf, _ = classify_intent_with_conf(query)
    RX_W = globals().get('RX_WEIGHT', 0.02)
    FUZZ_W = globals().get('FUZZ_WEIGHT', 0.01)
    ML_W = globals().get('ML_WEIGHT', 0.02)
    USE_ML = globals().get('USE_ML_BIAS', True)
    biased: List[tuple[Document, float]] = []
    for d, base in candidates:
        h = d.metadata.get('heading', '') or d.metadata.get('title', '')
        text_for_ml = f'{h}\n{d.page_content}'
        rx = RX_W * intent_conf * intent_bias(d.page_content, h, intent_label)
        fuzzb = FUZZ_W * heading_fuzzy_boost(query, h)
        ml = 0.0
        if USE_ML:
            try:
                scores = ml_scores_one_shot(text_for_ml, examples=EXAMPLES if 'EXAMPLES' in globals() else None)
                ml = ML_W * intent_conf * float(scores.get(intent_label, 0.0))
            except Exception as e:
                logger.warning('One-shot ML bias skipped: %s', e)
        biased.append((d, base + rx + fuzzb + ml))
    biased.sort(key=lambda x: x[1], reverse=True)
    seeds = [d for d, _ in biased[:k]]
    logger.info('Seeds selected: %d; neighbor window=%d', len(seeds), window_size)
    needs_defs = plan.need_definitions
    if needs_defs:
        has_defs = any((_looks_like_definitions(d) for d in seeds))
        if not has_defs:
            defs_doc = next((d for d, _ in candidates if _looks_like_definitions(d)), None)
            if defs_doc is not None:
                seeds = [defs_doc] + [d for d in seeds if d is not defs_doc]
                seeds = seeds[:k]
                logger.info('Definitions injected into seeds (from plan).')
    expanded: List[Document] = []
    seen: set = set()

    def add_neighbors(seed: Document):
        src = seed.metadata.get('source')
        sid = seed.metadata.get('section_id')
        if src is None or sid is None:
            sk = _key(seed)
            if sk not in seen:
                seen.add(sk)
                expanded.append(seed)
            return
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = sid
        order = DOC_ORDER.get(src, [])
        if not order:
            sk = _key(seed)
            if sk not in seen:
                seen.add(sk)
                expanded.append(seed)
            return
        for offset in range(-window_size, window_size + 1):
            neighbor_id = sid_int + offset if isinstance(sid_int, int) else sid_int
            doc = SECTION_INDEX.get((src, neighbor_id))
            if doc is None:
                continue
            sk = _key(doc)
            if sk not in seen:
                seen.add(sk)
                expanded.append(doc)
    for d in seeds:
        add_neighbors(d)
    final_docs = expanded[:max(k * (1 + 2 * window_size), k)]
    logger.info('Expanded final docs: %d (requested k=%d)', len(final_docs), k)
    if preview:
        serialized = serialize_preview(final_docs, max_chars_per_doc=preview_max_chars_per_doc, max_total_chars=preview_max_total_chars)
    else:
        serialized = '\n\n'.join((f'Source: {doc.metadata}\nContent: {doc.page_content}' for doc in final_docs))
    return (serialized, final_docs)


def _build_context_blocks(docs, *, max_context_chars: int=6000, max_chars_per_doc: int=900) -> Tuple[str, List[dict]]:
    """
    Turn retrieved docs into a compact, cited context string.
    Returns (context_text, refs) where refs carries id→metadata for later use.
    """
    blocks = []
    refs = []
    used = 0
    for idx, d in enumerate(docs, start=1):
        heading = d.metadata.get('heading') or d.metadata.get('title') or '<no heading>'
        src = d.metadata.get('source', '')
        sec = d.metadata.get('section_id', '')
        snippet = _squash_ws(d.page_content)[:max_chars_per_doc]
        block = f'[#{idx}] {heading}  (source={src}, section={sec})\n{snippet}\n'
        if used + len(block) > max_context_chars:
            break
        blocks.append(block)
        refs.append({'id': idx, 'heading': heading, 'source': src, 'section_id': sec, 'chars': len(snippet)})
        used += len(block)
    return ('\n---\n'.join(blocks), refs)


def generate_answer_rag(query: str, *, context: Optional[str]=None, docs: Optional[List[Document]]=None, k: int=6, window_size: int=1, use_mmr: bool=True, bm25_weight: float=0.0, rerank: bool=False, score_threshold: float=0.0, filters: Optional[dict]=None, max_context_chars: int=6000, max_chars_per_doc: int=900, **retrieve_kwargs) -> str:
    """
    Build/accept context, then ask the LLM.
    Priority:
      1) use provided `context` if given
      2) else build from provided `docs`
      3) else call `retrieve()` to get docs and build context
    """
    if context is not None:
        context_text = context.strip()
    else:
        if docs is None:
            _serialized, docs = retrieve(query, k=k, window_size=window_size, use_mmr=use_mmr, bm25_weight=bm25_weight, rerank=rerank, score_threshold=score_threshold, filters=filters, **retrieve_kwargs)
        if not docs:
            return 'I don’t know.'
        context_text, _refs = _build_context_blocks(docs, max_context_chars=max_context_chars, max_chars_per_doc=max_chars_per_doc)
    if not context_text.strip():
        return 'I don’t know.'
    system_prompt = f'You are a careful, citation-first assistant. Answer using ONLY the Context blocks below. If the answer is not present, reply exactly: “I don’t know based on the provided context.”\n\nContext blocks:\n{context_text}\n\nUser question: {query}\n\nRules:\n1) Use only information found in the Context blocks. Do NOT use outside knowledge.\n2) Cite evidence inline using [#id] immediately after each factual claim or number.    If multiple blocks support a claim, use comma-separated citations like [#2,#5].\n3) If you perform a calculation, show the equation with the sourced numbers and cite each number (e.g., A = 10 [#1] + 5 [#3]).\n4) Preserve terminology and section/heading names exactly as shown in the Context when referencing them.\n5) If blocks conflict, present the competing statements with their citations and say which one is more specific or recent if that is indicated in the Context; otherwise say the conflict cannot be resolved from the Context.\n6) If essential details are missing, state what is missing and answer: “I don’t know based on the provided context.”\n7) Be concise and clear. Do not speculate. Do not invent IDs or content.\n\nOutput format (Markdown):\n### Answer (with inline citations)\n- Brief direct answer in 1–3 sentences, citing the key block(s) (e.g., [#2]).\n\n- Short explanation with relevant sections/headings quoted or paraphrased, each claim cited.\n- Include brief quotes only when wording matters, in quotes with citations (e.g., "…" [#3]).\n\n###Risk Assesment\n- List all the risk that may apply in the contract with citations (e.g., "…" [#3])\n\n### Gaps or Conflicts (if any)\n- Note missing info or contradictions and cite the relevant blocks.\n'
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': query}]
    resp = client.chat.complete(model='mistral-large-latest', messages=messages, temperature=0.1)
    return resp.choices[0].message.content


# ---- Executable cells ----


%pip install -U langchain langchain-community langchain-mistralai mistralai \
  sentence-transformers rapidfuzz pymupdf langid rank-bm25


!pip install -q huggingface_hub


if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = userdata.get('MISTRAL_API_KEY')


llm = init_chat_model("mistral-large-latest", model_provider="mistralai")


pip install langchain_huggingface


hf = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


print("OpenAI key:", bool(os.getenv("MISTRAL_API_KEY")))


vector_store = InMemoryVectorStore(hf)


pdf_path = get_pdf_via_upload()


print(f"✅ Using PDF: {pdf_path}")


if not pdf_path.lower().endswith(".pdf"):
    raise ValueError("Error: Please provide a .pdf file")


_ = outputcom(pdf_path, debug=True, print_pages=True, max_pages=2)


text, diag = outputcom(pdf_path, debug=True, return_stats=True)


for s in diag:
    print(s)


_SINGLE_NUM_DOT = re.compile(r'^\s*(?P<num>\d+)\.\s+(?P<title>.+?)\s*$')


_BOLD_LINES_CACHE: Dict[str, set] = {}


SECTION_INDEX: Dict[Tuple[str, int], "Document"] = {}


DOC_ORDER: Dict[str, List[int]] = defaultdict(list)


preview_splits(text, max_preview_chars=300)


doc_id = make_source_id(pdf_path, version="2023-06-16", content_text=text)


docs = build_documents_from_text(text, pdf_path=pdf_path, doc_id=doc_id)


chunk_ids = [f"{doc_id}__s{d.metadata['section_id']}" for d in docs]


vector_store.add_documents(docs, ids=chunk_ids)


bm25 = build_bm25_retriever(docs, k=100)


RIGHT_SIGNS_RX = [re.compile(p, re.I) for p in [
    r"\bright to\b", r"\bmay\b", r"\bpermitted\b", r"\bentitled\b", r"\bauthoriz"
]]


OBLIG_SIGNS_RX = [re.compile(p, re.I) for p in [
    r"\bshall\b", r"\bmust\b", r"\brequired to\b", r"\bduty\b", r"\bresponsib"
]]


PROH_SIGNS_RX  = [re.compile(p, re.I) for p in [
    r"\bshall not\b", r"\bmay not\b", r"\bprohibit", r"\brestrict"
]]


RISK_SIGNS_RX  = [re.compile(p, re.I) for p in [
    r"\bindemnif(y|ication|ies)\b", r"\bhold harmless\b", r"\bliabilit(y|ies)\b",
    r"\blimitation of liability\b", r"\bliability cap\b|\bcap on liability\b",
    r"\b(exclude|exclusion)s?\b.*\bdamages\b", r"\bdamages\b",
    r"\bconsequential damages\b|\bindirect\b.*\bdamages\b|\bspecial\b.*\bdamages\b|\bpunitiv(e)? damages\b",
    r"\brisk\b", r"\bloss(es)?\b", r"\bdisclaimer\b|\bwarranty disclaimer\b"
]]


_CLASS_SEEDS = {
    "RIGHTS":       "right to may permitted entitled authorization license",
    "OBLIGATIONS":  "shall must required obligation duty responsible",
    "PROHIBITIONS": "shall not may not prohibited restriction no",
    "RISK":         "liability indemnify hold harmless damages limitation cap exclude warranty disclaimer"
}


USE_ML_BIAS = True


ML_WEIGHT   = 0.02


RX_WEIGHT   = 0.02


FUZZ_WEIGHT = 0.01


EXAMPLES: List[Tuple[str, Dict[str, float]]] = [
    ("The Supplier shall deliver within 7 days.", {"RIGHTS":0.0, "OBLIGATIONS":0.95, "PROHIBITIONS":0.0, "RISK":0.05}),
    ("If required in the Contract, Supplier must give warranty to Purchaser on the Goods and/or Services from any damage",
     {"RIGHTS":0.0, "OBLIGATIONS":0.95, "PROHIBITIONS":0.0, "RISK":0.05}),
    (". Neither party may assign any of its rights or obligations hereunder, whether by operation of law or otherwise, without the prior written consent of the other party (not to be unreasonably withheld).",
     {"RIGHTS":0.95, "OBLIGATIONS":0.0, "PROHIBITIONS":0.0, "RISK":0.05}),
    (" A party may immediately terminate this Agreement for cause: (i) upon 30 days written notice of a material breach...",
     {"RIGHTS":0.95, "OBLIGATIONS":0.0, "PROHIBITIONS":0.0, "RISK":0.65})
]


LABELS = ["RIGHTS","OBLIGATIONS","PROHIBITIONS","RISK"]


logger = logging.getLogger(__name__)


_CAP_STOP = {"Agreement","Schedule","Exhibit","Section","Party","Parties","Company"}


_SYNONYMS = {
    r"\bterminate\b": ["termination", "rescind", "cancel", "end", "expire"],
    r"\brenew\b":     ["renewal", "extend", "extension", "prolong"],
    r"\bliability\b": ["indemnify", "indemnification", "damages"],
    r"\bconfidential\b": ["non-disclosure", "NDA", "trade secret"],
}


_LEGALBERT_CACHE = {}


logger = logging.getLogger("rag.retrieve")


if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)


logger.setLevel(logging.INFO)


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CE = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE)


client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


input_message = input("Enter your question: ").strip() or "-"


serialized, docs = retrieve(
    input_message,
    k=6,
    window_size=1,           # neighbor expansion window (±1 section)
    fetch_k=60,              # candidate pool size before fusion/rerank
    bm25_retriever=bm25,     # REQUIRED: BM25 is always used
    bm25_weight=0.4,         # fusion weight between vector and BM25
    query_prefix="query:",   # helpful for asymmetric embedding models
)


print("=== Retrieved Context ===")


print(serialized)


print()


answer = generate_answer_rag(input_message, docs=docs)


print("=== Assistant Answer ===")


print(answer)