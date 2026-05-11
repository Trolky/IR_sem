import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Iterable, Set


def tf_raw(tokens: Sequence[str]) -> Dict[str, float]:
    """Compute raw term frequency (TF) as token occurrence counts.

    Args:
        tokens: Sequence of tokens (e.g., a list of strings).

    Returns:
        A mapping term -> occurrence count (float for convenience in later math).
    """
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    return tf


def tf_weighted_log(tokens: Sequence[str]) -> Dict[str, float]:
    """Compute weighted term frequency: 1 + log10(tf).

    This is a common TF variant that dampens the impact of very frequent terms
    within a document/query.

    Notes:
        - Terms with tf=0 don't appear in the output.
        - For tf >= 1, weight is >= 1.
    """
    raw = tf_raw(tokens)
    return {t: 1.0 + math.log10(c) for t, c in raw.items() if c > 0.0}


def l2_norm(vec: Dict[str, float]) -> float:
    """Return the L2 norm (Euclidean length) of a sparse vector.

    Args:
        vec: Sparse vector as dict term -> weight.

    Returns:
        The L2 norm: sqrt(sum_i v_i^2). Returns 0.0 for an empty vector.
    """
    return math.sqrt(sum(v * v for v in vec.values()))


def l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    """Return an L2-normalized copy of a sparse vector.

    Args:
        vec: Sparse vector as dict term -> weight.

    Returns:
        A new dict term -> weight / ||vec||_2.
        If the norm is 0.0 (empty / all-zero vector), returns {}.
    """
    n = l2_norm(vec)
    if n == 0.0:
        return {}
    return {k: v / n for k, v in vec.items()}


@dataclass
class SearchResult:
    """Search result item.

    Attributes:
        doc_id: Document identifier (e.g., "d1" or a URL).
        score: Cosine similarity score (higher = more relevant).
    """

    doc_id: str
    score: float


class TfidfIndex:
    """A simple TF-IDF index with cosine similarity.

    Conventions:

    - TF = raw counts (number of occurrences of a term in a document)
    - DF(term) = number of documents containing the term at least once
    - IDF(term) = log10(N / DF(term))
    - TF-IDF(term, doc) = TF(term, doc) * IDF(term)
    - Similarity = cosine(query_tfidf, doc_tfidf)
    """

    def __init__(self):
        """Create an empty index."""
        self._doc_ids: List[str] = []
        self._doc_tfidf: List[Dict[str, float]] = []
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._N: int = 0

        # Inverted index over TF-IDF weights.
        # term -> {doc_index -> tfidf_weight}
        self._postings: Dict[str, Dict[int, float]] = {}

        # Precomputed L2 norms for document TF-IDF vectors (same order as _doc_ids).
        self._doc_norms: List[float] = []

    @property
    def N(self) -> int:
        """Number of indexed documents."""
        return self._N

    @property
    def df(self) -> Dict[str, int]:
        """Document frequency for each term (returns a copy)."""
        return dict(self._df)

    @property
    def idf(self) -> Dict[str, float]:
        """Inverse document frequency for each term (returns a copy)."""
        return dict(self._idf)

    def build(self, documents: Sequence[Tuple[str, Sequence[str]]]) -> None:
        """Build DF/IDF statistics and TF-IDF vectors for all documents.

        Args:
            documents: Sequence of (doc_id, tokens) pairs.
                - doc_id: arbitrary string identifier
                - tokens: tokenized document content

        Returns:
            None. Results are stored in the instance (df/idf and document vectors).

        Notes:
            An empty collection is allowed (N=0, empty df/idf).
        """
        self._doc_ids = []
        self._doc_tfidf = []
        self._df = {}
        self._idf = {}
        self._postings = {}
        self._doc_norms = []
        self._N = len(documents)

        # 1) compute df
        doc_tfs: List[Dict[str, float]] = []
        for doc_id, tokens in documents:
            self._doc_ids.append(doc_id)
            tf = tf_weighted_log(tokens)
            doc_tfs.append(tf)
            for term in tf.keys():
                self._df[term] = self._df.get(term, 0) + 1

        # 2) compute idf
        for term, df in self._df.items():
            self._idf[term] = math.log10(self._N / df) if self._N else 0.0

        # 3) compute doc tf-idf + inverted index
        for doc_i, tf in enumerate(doc_tfs):
            vec: Dict[str, float] = {}
            for term, c in tf.items():
                w = c * self._idf.get(term, 0.0)
                if w == 0.0:
                    continue
                vec[term] = w
                bucket = self._postings.get(term)
                if bucket is None:
                    bucket = {}
                    self._postings[term] = bucket
                bucket[doc_i] = w
            self._doc_tfidf.append(vec)
            self._doc_norms.append(l2_norm(vec))

    def vectorize_query(self, query_tokens: Sequence[str], normalize: bool = False) -> Dict[str, float]:
        """Convert query tokens into a TF-IDF vector using the index IDF.

        Terms not present in the index vocabulary (i.e., without IDF) are ignored.

        Args:
            query_tokens: Query tokens.
            normalize: If True, returns an L2-normalized TF-IDF vector.

        Returns:
            Query TF-IDF sparse vector as dict term -> weight.
            If no query term is in the vocabulary, returns {}.
        """
        q_tf = tf_weighted_log(query_tokens)
        q_vec: Dict[str, float] = {}
        for term, c in q_tf.items():
            if term in self._idf:
                q_vec[term] = c * self._idf[term]
        return l2_normalize(q_vec) if normalize else q_vec

    def _candidate_doc_indexes(self, query_terms: Iterable[str]) -> Set[int]:
        """Return candidate document indexes (union of postings for query terms)."""
        cand: Set[int] = set()
        for t in query_terms:
            p = self._postings.get(t)
            if p:
                cand.update(p.keys())
        return cand

    def search(self, query_tokens: Sequence[str], k: int = 5, normalize: bool = True) -> List[SearchResult]:
        """Return top-k documents by cosine similarity to the query.

        This version uses an inverted index so it only scores documents that
        contain at least one in-vocabulary query term.

        Args:
            query_tokens: Query tokens.
            k: Number of results to return.
            normalize: If True, explicitly L2-normalizes both query and document
                TF-IDF vectors, and then computes cosine on unit vectors.
                This yields the same final score as cosine on raw TF-IDF, but
                makes intermediate "normalized tf-idf" weights explicit.

        Returns:
            List of `SearchResult` sorted by descending score.
            If the query is empty / out-of-vocabulary, scores will be 0.0.
        """
        # Build query tf-idf (optionally normalized).
        q_vec = self.vectorize_query(query_tokens, normalize=normalize)

        # If query is empty or entirely OOV, return top-k zeros (stable order).
        if not q_vec:
            return [SearchResult(doc_id=doc_id, score=0.0) for doc_id in self._doc_ids[:k]]

        # Candidate docs = union of postings for query terms.
        candidates = self._candidate_doc_indexes(q_vec.keys())
        if not candidates:
            return [SearchResult(doc_id=doc_id, score=0.0) for doc_id in self._doc_ids[:k]]

        scored: List[SearchResult] = []

        if normalize:
            # With normalized vectors, cosine is just dot on overlapping terms.
            # We'll accumulate dot products from postings directly.
            acc: Dict[int, float] = {}
            for term, q_w in q_vec.items():
                postings = self._postings.get(term)
                if not postings:
                    continue
                for doc_i, d_w in postings.items():
                    acc[doc_i] = acc.get(doc_i, 0.0) + q_w * (d_w / self._doc_norms[doc_i] if self._doc_norms[doc_i] else 0.0)

            for doc_i, score in acc.items():
                scored.append(SearchResult(doc_id=self._doc_ids[doc_i], score=score))
        else:
            # Raw cosine: dot / (||q|| * ||d||)
            q_norm = l2_norm(q_vec)
            if q_norm == 0.0:
                return [SearchResult(doc_id=doc_id, score=0.0) for doc_id in self._doc_ids[:k]]

            acc: Dict[int, float] = {}
            for term, q_w in q_vec.items():
                postings = self._postings.get(term)
                if not postings:
                    continue
                for doc_i, d_w in postings.items():
                    acc[doc_i] = acc.get(doc_i, 0.0) + q_w * d_w

            for doc_i, dot in acc.items():
                d_norm = self._doc_norms[doc_i]
                score = dot / (q_norm * d_norm) if d_norm else 0.0
                scored.append(SearchResult(doc_id=self._doc_ids[doc_i], score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

def pretty_vec(vec: Dict[str, float]) -> str:
    """Pretty-print a sparse vector (sorted terms, 3 decimal places).

    Args:
        vec: Sparse vector as dict term -> weight.

    Returns:
        A formatted string like "{term: 0.123, ...}".
    """
    items = sorted(vec.items())
    return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in items) + "}"
