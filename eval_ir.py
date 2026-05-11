import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from preprocessing import WowheadPreprocessor
from tfidf_search import TfidfIndex


@dataclass(frozen=True)
class QueryItem:
    """Query item from the evaluation dataset."""

    qid: str
    text: str
    relevant_doc_ids: Tuple[str, ...]


def load_eval_documents(path: str) -> List[Dict]:
    """Load evaluation documents JSON (a JSON array of objects)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_queries(path: str) -> List[QueryItem]:
    """Load evaluation queries JSON and extract qrels from evidence_list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[QueryItem] = []
    for rec in data:
        qid = str(rec.get("id"))
        text = rec.get("description") or rec.get("query") or ""

        ev = rec.get("evidence_list") or []
        rel: List[str] = []
        for e in ev:
            did = e.get("id")
            if did is not None:
                rel.append(str(did))

        # Deduplicate but keep stable order
        dedup = list(dict.fromkeys(rel).keys())
        out.append(QueryItem(qid=qid, text=text, relevant_doc_ids=tuple(dedup)))
    return out


def average_precision(ranked_doc_ids: Sequence[str], relevant: Iterable[str], k: Optional[int] = None) -> float:
    """Compute Average Precision (AP) for one query."""
    rel_set = set(relevant)
    if not rel_set:
        return 0.0

    if k is not None:
        ranked_doc_ids = ranked_doc_ids[:k]

    hit = 0
    s = 0.0
    for i, did in enumerate(ranked_doc_ids, start=1):
        if did in rel_set:
            hit += 1
            s += hit / i
    return s / len(rel_set)


def mean_average_precision(run: Dict[str, List[str]], qrels: Dict[str, Iterable[str]], k: Optional[int] = None) -> float:
    """Compute MAP over queries."""
    aps: List[float] = []
    for qid, rel in qrels.items():
        ranked = run.get(qid, [])
        aps.append(average_precision(ranked, rel, k=k))
    return float(np.mean(aps)) if aps else 0.0


class LsaIndex:
    """LSA/LSI index built from a sparse TF-IDF matrix using truncated SVD."""

    def __init__(self, k: int = 200):
        self.k = k
        self._pre = WowheadPreprocessor(use_lemmatization=True, remove_diacritics=True)

        self._doc_ids: List[str] = []
        self._tfidf: Optional[TfidfIndex] = None

        self._U: Optional[np.ndarray] = None
        self._S: Optional[np.ndarray] = None
        self._VT: Optional[np.ndarray] = None
        self._term_to_i: Dict[str, int] = {}

    @property
    def doc_ids(self) -> List[str]:
        return list(self._doc_ids)

    def build(self, documents: Sequence[Tuple[str, Sequence[str]]]) -> None:
        tfidf = TfidfIndex()
        tfidf.build(documents)
        self._tfidf = tfidf

        terms = list(tfidf.idf.keys())
        self._term_to_i = {t: i for i, t in enumerate(terms)}
        self._doc_ids = [doc_id for doc_id, _ in documents]

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        for j, (_, tokens) in enumerate(documents):
            tf: Dict[str, int] = {}
            for t in tokens:
                if t in self._term_to_i:
                    tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                w_tf = 1.0 + math.log10(c)
                w = w_tf * tfidf.idf.get(t, 0.0)
                if w == 0.0:
                    continue
                rows.append(self._term_to_i[t])
                cols.append(j)
                vals.append(w)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(len(terms), len(documents)), dtype=np.float64)

        if A.shape[0] == 0 or A.shape[1] == 0:
            self._U = np.zeros((A.shape[0], 0), dtype=np.float64)
            self._S = np.zeros((0,), dtype=np.float64)
            self._VT = np.zeros((0, A.shape[1]), dtype=np.float64)
            return

        k = min(self.k, min(A.shape) - 1)
        if k <= 0:
            k = 1

        U, S, VT = svds(A, k=k)

        order = np.argsort(S)[::-1]
        self._U = U[:, order]
        self._S = S[order]
        self._VT = VT[order, :]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._tfidf is None or self._U is None or self._S is None or self._VT is None:
            return []

        q_tokens = self._pre.clean_text(query).split()
        if not q_tokens:
            return []

        q_vec = np.zeros((len(self._term_to_i),), dtype=np.float64)
        tf: Dict[str, int] = {}
        for t in q_tokens:
            if t in self._term_to_i:
                tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            w_tf = 1.0 + math.log10(c)
            q_vec[self._term_to_i[t]] = w_tf * self._tfidf.idf.get(t, 0.0)

        if np.all(q_vec == 0):
            return []

        q_lat = (q_vec @ self._U) / self._S
        doc_lat = (self._VT.T * self._S)

        qn = float(np.linalg.norm(q_lat))
        dn = np.linalg.norm(doc_lat, axis=1)
        denom = dn * qn
        denom[denom == 0] = np.inf
        scores = (doc_lat @ q_lat) / denom

        best = np.argsort(scores)[::-1][:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in best]


def build_documents_for_index(raw_docs: List[Dict]) -> List[Tuple[str, List[str]]]:
    """Prepare tokenized docs (doc_id, tokens) for TF-IDF/LSA."""
    pre = WowheadPreprocessor(use_lemmatization=True, remove_diacritics=True)
    docs: List[Tuple[str, List[str]]] = []
    for i, d in enumerate(raw_docs):
        doc_id = str(d.get("id") or d.get("doc_id") or f"doc_{i}")
        text = " ".join(filter(None, [d.get("title", ""), d.get("content", ""), d.get("text", "")]))
        tokens = pre.clean_text(text).split()
        if tokens:
            docs.append((doc_id, tokens))
    return docs


def evaluate_tfidf(documents: Sequence[Tuple[str, Sequence[str]]], queries: Sequence[QueryItem], top_k: int = 20) -> Dict[str, List[str]]:
    idx = TfidfIndex()
    idx.build(documents)
    pre = WowheadPreprocessor(use_lemmatization=True, remove_diacritics=True)

    run: Dict[str, List[str]] = {}
    for q in queries:
        q_tokens = pre.clean_text(q.text).split()
        results = idx.search(q_tokens, k=top_k, normalize=True)
        run[q.qid] = [r.doc_id for r in results]
    return run

