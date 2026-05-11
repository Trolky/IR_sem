import heapq
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def tokenize_boolean_query(query: str) -> List[str]:
    """Tokenize a boolean query.

    Supports operators (case-insensitive): AND, OR, NOT, and parentheses.

    Examples:
        "apple AND (banana OR cherry) NOT date"

    Returns:
        A list of tokens in order.
    """
    if not query:
        return []

    # Parentheses as standalone tokens, operators, or a term.
    pattern = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")
    toks = pattern.findall(query)
    out: List[str] = []
    for t in toks:
        u = t.upper()
        if u in {"AND", "OR", "NOT"}:
            out.append(u)
        elif t in {"(", ")"}:
            out.append(t)
        else:
            out.append(t)
    return out


@dataclass(frozen=True)
class _Node:
    op: str  # TERM, AND, OR, NOT
    value: Optional[str] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class BooleanQueryParseError(ValueError):
    """Raised when the boolean query cannot be parsed."""


def _to_rpn(tokens: Sequence[str]) -> List[str]:
    """Convert an infix boolean expression to RPN using the shunting-yard algorithm."""
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    assoc = {"NOT": "right", "AND": "left", "OR": "left"}

    output: List[str] = []
    stack: List[str] = []

    def is_op(tok: str) -> bool:
        return tok in ("NOT", "AND", "OR")

    for tok in tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack or stack[-1] != "(":
                raise BooleanQueryParseError("Mismatched parentheses in query")
            stack.pop()  # remove '('
        elif is_op(tok):
            while stack and is_op(stack[-1]):
                top = stack[-1]
                if (assoc[tok] == "left" and prec[tok] <= prec[top]) or (assoc[tok] == "right" and prec[tok] < prec[top]):
                    output.append(stack.pop())
                else:
                    break
            stack.append(tok)
        else:
            # term
            output.append(tok)

    while stack:
        top = stack.pop()
        if top in ("(", ")"):
            raise BooleanQueryParseError("Mismatched parentheses in query")
        output.append(top)

    return output


def _rpn_to_ast(rpn: Sequence[str]) -> _Node:
    stack: List[_Node] = []
    for tok in rpn:
        if tok == "NOT":
            if not stack:
                raise BooleanQueryParseError("NOT without an operand")
            a = stack.pop()
            stack.append(_Node(op="NOT", left=a))
        elif tok in ("AND", "OR"):
            if len(stack) < 2:
                raise BooleanQueryParseError(f"{tok} without two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(_Node(op=tok, left=a, right=b))
        else:
            stack.append(_Node(op="TERM", value=tok))

    if len(stack) != 1:
        raise BooleanQueryParseError("Invalid query structure")
    return stack[0]


def parse_boolean_query(query: str) -> _Node:
    toks = tokenize_boolean_query(query)
    if not toks:
        raise BooleanQueryParseError("Empty query")
    rpn = _to_rpn(toks)
    return _rpn_to_ast(rpn)


class BooleanIndex:
    """Simple boolean inverted index (term -> set(doc_id)).

    - Indexes pre-tokenized documents (already normalized/lowercased/lemmatized as you prefer).
    - Supports boolean queries with AND/OR/NOT and parentheses.

    Notes:
        - Terms are matched exactly as provided. You should normalize query terms
          the same way you normalize document tokens.
        - NOT uses the universe of all doc_ids currently present in the index.

    Optimizations over the baseline:
        1. AND NOT → set difference instead of materializing the full NOT set.
        2. Short-circuit AND: returns immediately when the left operand is empty.
        3. Short-circuit OR: returns immediately when the left operand equals _all_docs.
        4. Operand reordering for AND: always intersects smaller set with larger set.
    """

    def __init__(self):
        self._postings: Dict[str, frozenset] = {}
        self._all_docs: Set[str] = set()

    @property
    def all_docs(self) -> Set[str]:
        return set(self._all_docs)

    def add_documents(self, documents: Iterable[Tuple[str, Sequence[str]]]) -> None:
        # Collect new postings into mutable sets, then freeze them at the end.
        pending: Dict[str, Set[str]] = {}
        for doc_id, tokens in documents:
            self._all_docs.add(doc_id)
            seen: Set[str] = set()
            for t in tokens:
                if t in seen:
                    continue
                seen.add(t)
                pending.setdefault(t, set()).add(doc_id)

        # Merge pending into existing postings and re-freeze.
        for term, new_docs in pending.items():
            existing = self._postings.get(term)
            if existing is None:
                self._postings[term] = frozenset(new_docs)
            else:
                self._postings[term] = existing | new_docs

    def evaluate(self, query: str) -> Set[str]:
        ast = parse_boolean_query(query)
        return self._eval_node(ast)

    def _eval_node(self, node: _Node) -> Set[str]:
        if node.op == "TERM":
            assert node.value is not None
            return self._postings.get(node.value, frozenset())

        if node.op == "NOT":
            assert node.left is not None
            return set(self._all_docs) - self._eval_node(node.left)

        if node.op == "AND":
            assert node.left is not None and node.right is not None

            # Opt 1: A AND (NOT B)  →  A - B  (avoids materializing _all_docs - B)
            if node.right.op == "NOT":
                assert node.right.left is not None
                return self._eval_node(node.left) - self._eval_node(node.right.left)
            if node.left.op == "NOT":
                assert node.left.left is not None
                return self._eval_node(node.right) - self._eval_node(node.left.left)

            left = self._eval_node(node.left)

            # Opt 2: short-circuit — empty left means intersection is always empty.
            if not left:
                return set()

            right = self._eval_node(node.right)

            if len(left) > len(right):
                left, right = right, left
            return left & right

        if node.op == "OR":
            assert node.left is not None and node.right is not None

            left = self._eval_node(node.left)

            # Opt 3: short-circuit — if left already covers everything, OR adds nothing.
            if left == self._all_docs:
                return set(self._all_docs)

            return left | self._eval_node(node.right)

        raise BooleanQueryParseError(f"Unknown node op: {node.op}")

    def search(self, query: str, k: Optional[int] = None) -> List[str]:
        """Evaluate a boolean query and return matching doc_ids.

        Args:
            query: Boolean query string.
            k: Optional limit for number of returned doc_ids.

        Returns:
            Sorted list of matching doc_ids (lexicographically). Sorting provides
            stable output (boolean retrieval itself has no ranking).
        """
        hits = self.evaluate(query)
        if k is None:
            return sorted(hits)
        return heapq.nsmallest(k, hits)