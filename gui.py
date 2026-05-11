#!/usr/bin/env python3
"""
IR System GUI — Automatická indexace a vyhledávání dokumentů.
Podporuje Wowhead articles (JSONL) a evaluační data (JSON).
"""

import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional, Tuple

from boolean_search import BooleanIndex, BooleanQueryParseError, tokenize_boolean_query
from eval_ir import (
    LsaIndex,
    QueryItem,
    load_eval_documents,
    load_eval_queries,
    mean_average_precision,
)
from preprocessing import WowheadPreprocessor
from tfidf_search import TfidfIndex

WOWHEAD_PATH = "wowhead_articles.jsonl"
EVAL_DOCS_PATH = "data/documents.json"
EVAL_QUERIES_PATH = "data/full_text_queries.json"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_wowhead_articles(path: str) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "id" not in d:
                d["id"] = d.get("url") or f"wh_{i}"
            docs.append(d)
    return docs


def build_index_from_raw(
    raw_docs: List[Dict], preprocessor: WowheadPreprocessor
) -> Tuple[TfidfIndex, BooleanIndex, Dict[str, Dict], List[Tuple[str, List[str]]]]:
    doc_lookup: Dict[str, Dict] = {}
    doc_tuples: List[Tuple[str, List[str]]] = []

    for i, d in enumerate(raw_docs):
        doc_id = str(d.get("id") or d.get("url") or f"doc_{i}")
        text = " ".join(
            filter(None, [d.get("title", ""), d.get("content", ""), d.get("text", "")])
        )
        tokens = preprocessor.clean_text(text).split()
        if tokens:
            doc_tuples.append((doc_id, tokens))
            doc_lookup[doc_id] = d

    tfidf = TfidfIndex()
    tfidf.build(doc_tuples)

    boolean_idx = BooleanIndex()
    boolean_idx.add_documents(doc_tuples)

    return tfidf, boolean_idx, doc_lookup, doc_tuples


def preprocess_boolean_query_terms(query: str, preprocessor: WowheadPreprocessor) -> str:
    """Lowercase and stem/lemmatize term tokens inside a boolean query string."""
    tokens = tokenize_boolean_query(query)
    result = []
    for tok in tokens:
        if tok in {"AND", "OR", "NOT", "(", ")"}:
            result.append(tok)
        else:
            cleaned = preprocessor.clean_text(tok).split()
            result.append(cleaned[0] if cleaned else tok.lower())
    return " ".join(result)


# ---------------------------------------------------------------------------
# Thread-safe log mixin
# ---------------------------------------------------------------------------

class LogMixin:
    def _init_log(self, log_widget: scrolledtext.ScrolledText, root: tk.Tk) -> None:
        self._log_widget = log_widget
        self._root = root
        self._log_queue: queue.Queue = queue.Queue()
        self._drain_log()

    def _drain_log(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._log_widget.config(state="normal")
                self._log_widget.insert("end", msg + "\n")
                self._log_widget.see("end")
                self._log_widget.config(state="disabled")
        except queue.Empty:
            pass
        self._root.after(50, self._drain_log)

    def _log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def _clear_log_widget(self) -> None:
        self._log_widget.config(state="normal")
        self._log_widget.delete("1.0", "end")
        self._log_widget.config(state="disabled")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class IRApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("IR System — Automatická indexace a vyhledávání")
        self.root.geometry("1200x800")
        self.root.minsize(950, 650)

        # name → {tfidf, boolean, doc_lookup, preprocessor, raw_docs, doc_tuples}
        self.indexes: Dict[str, Dict] = {}

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_index = IndexTab(nb, self)
        self.tab_search = SearchTab(nb, self)
        self.tab_eval = EvalTab(nb, self)

        nb.add(self.tab_index.frame, text="  Indexace  ")
        nb.add(self.tab_search.frame, text="  Vyhledávání  ")
        nb.add(self.tab_eval.frame, text="  Evaluace  ")

    def register_index(
        self,
        name: str,
        tfidf: TfidfIndex,
        boolean: BooleanIndex,
        doc_lookup: Dict,
        preprocessor: WowheadPreprocessor,
        raw_docs: List[Dict],
        doc_tuples: List[Tuple[str, List[str]]],
    ) -> None:
        self.indexes[name] = {
            "tfidf": tfidf,
            "boolean": boolean,
            "doc_lookup": doc_lookup,
            "preprocessor": preprocessor,
            "raw_docs": raw_docs,
            "doc_tuples": doc_tuples,
        }
        self.root.after(0, self.tab_search.refresh_index_combo)
        self.root.after(0, self.tab_eval.refresh_index_combo)
        self.root.after(0, self.tab_index.refresh_add_combo)


# ---------------------------------------------------------------------------
# Tab: Indexace
# ---------------------------------------------------------------------------

class IndexTab(LogMixin):
    def __init__(self, parent: ttk.Notebook, app: IRApp) -> None:
        self.app = app
        self.frame = ttk.Frame(parent)
        self._build_widgets()

    def _build_widgets(self) -> None:
        P = {"padx": 8, "pady": 4}

        # Source selection
        src_lf = ttk.LabelFrame(self.frame, text="Zdroj dat")
        src_lf.pack(fill="x", **P)

        self.src_var = tk.StringVar(value="wowhead")
        rb_row = ttk.Frame(src_lf)
        rb_row.pack(fill="x", padx=6, pady=4)

        ttk.Radiobutton(rb_row, text="Wowhead articles  (wowhead_articles.jsonl)",
                        variable=self.src_var, value="wowhead",
                        command=self._on_src).pack(side="left", padx=6)
        ttk.Radiobutton(rb_row, text="Evaluační data  (data/documents.json)",
                        variable=self.src_var, value="eval",
                        command=self._on_src).pack(side="left", padx=6)
        ttk.Radiobutton(rb_row, text="Vlastní soubor:",
                        variable=self.src_var, value="custom",
                        command=self._on_src).pack(side="left", padx=6)

        file_row = ttk.Frame(src_lf)
        file_row.pack(fill="x", padx=6, pady=(0, 4))
        self.custom_path_var = tk.StringVar()
        self.custom_entry = ttk.Entry(file_row, textvariable=self.custom_path_var,
                                      state="disabled", width=55)
        self.custom_entry.pack(side="left", padx=2)
        self.browse_btn = ttk.Button(file_row, text="Procházet…",
                                     command=self._browse, state="disabled")
        self.browse_btn.pack(side="left", padx=4)
        ttk.Label(file_row, text="(JSONL = wowhead formát  |  JSON array = evaluační formát)",
                  foreground="gray", font=("Segoe UI", 8)).pack(side="left", padx=6)

        # Index name
        name_lf = ttk.LabelFrame(self.frame, text="Název indexu")
        name_lf.pack(fill="x", **P)
        name_row = ttk.Frame(name_lf)
        name_row.pack(fill="x", padx=6, pady=4)
        self.index_name_var = tk.StringVar(value="wowhead")
        ttk.Entry(name_row, textvariable=self.index_name_var, width=28).pack(side="left")
        ttk.Label(name_row,
                  text=" ← Pod tímto názvem bude index dostupný ve vyhledávání a evaluaci",
                  foreground="gray", font=("Segoe UI", 8)).pack(side="left")

        # Preprocessing options
        pre_lf = ttk.LabelFrame(self.frame, text="Předzpracování textu")
        pre_lf.pack(fill="x", **P)
        pre_row = ttk.Frame(pre_lf)
        pre_row.pack(fill="x", padx=6, pady=4)

        self.lemma_var = tk.BooleanVar(value=True)
        self.stem_var = tk.BooleanVar(value=False)
        self.diac_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(pre_row, text="Lemmatizace (WordNet)",
                        variable=self.lemma_var, command=self._on_lemma).pack(side="left", padx=8)
        ttk.Checkbutton(pre_row, text="Stemming (Porter)",
                        variable=self.stem_var, command=self._on_stem).pack(side="left", padx=8)
        ttk.Checkbutton(pre_row, text="Odstranění diakritiky",
                        variable=self.diac_var).pack(side="left", padx=8)
        ttk.Label(pre_row,
                  text="   • Stopwords se vždy odstraňují  • Lemmatizace ⊕ Stemming (vzájemně se vylučují)",
                  foreground="gray", font=("Segoe UI", 8)).pack(side="left")

        # Build button row
        btn_row = ttk.Frame(self.frame)
        btn_row.pack(fill="x", **P)
        self.build_btn = ttk.Button(btn_row, text="⚙  Vytvořit index",
                                    command=self._start_build, width=18)
        self.build_btn.pack(side="left")
        self.progress = ttk.Progressbar(btn_row, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=10)
        self.status_lbl = ttk.Label(btn_row, text="", foreground="#1a7a1a", font=("Segoe UI", 9, "bold"))
        self.status_lbl.pack(side="left")

        # Doindexování
        add_lf = ttk.LabelFrame(self.frame, text="Doindexovat — přidat data do existujícího indexu")
        add_lf.pack(fill="x", **P)

        add_r1 = ttk.Frame(add_lf)
        add_r1.pack(fill="x", padx=6, pady=4)
        ttk.Label(add_r1, text="Cílový index:").pack(side="left")
        self.add_target_var = tk.StringVar()
        self.add_target_combo = ttk.Combobox(add_r1, textvariable=self.add_target_var,
                                             width=18, state="readonly")
        self.add_target_combo.pack(side="left", padx=(2, 16))
        ttk.Label(add_r1, text="Soubor s novými daty:").pack(side="left")
        self.add_path_var = tk.StringVar()
        ttk.Entry(add_r1, textvariable=self.add_path_var, width=38).pack(side="left", padx=(2, 2))
        ttk.Button(add_r1, text="…", width=3,
                   command=lambda: self.add_path_var.set(
                       filedialog.askopenfilename(
                           filetypes=[("JSON / JSONL", "*.json *.jsonl"), ("Všechny", "*.*")]
                       ) or self.add_path_var.get()
                   )).pack(side="left")

        add_r2 = ttk.Frame(add_lf)
        add_r2.pack(fill="x", padx=6, pady=(0, 4))
        self.add_btn = ttk.Button(add_r2, text="Přidat dokumenty", command=self._start_add, width=20)
        self.add_btn.pack(side="left")
        self.add_status_lbl = ttk.Label(add_r2, text="", foreground="#1a7a1a")
        self.add_status_lbl.pack(side="left", padx=8)

        # Log area
        log_lf = ttk.LabelFrame(self.frame, text="Průběh a statistiky")
        log_lf.pack(fill="both", expand=True, **P)
        log_widget = scrolledtext.ScrolledText(log_lf, height=10, state="disabled",
                                               font=("Consolas", 9))
        log_widget.pack(fill="both", expand=True, padx=4, pady=4)
        self._init_log(log_widget, self.app.root)

    def _on_src(self) -> None:
        v = self.src_var.get()
        st = "normal" if v == "custom" else "disabled"
        self.custom_entry.config(state=st)
        self.browse_btn.config(state=st)
        names = {"wowhead": "wowhead", "eval": "eval"}
        if v in names:
            self.index_name_var.set(names[v])

    def _on_lemma(self) -> None:
        if self.lemma_var.get():
            self.stem_var.set(False)

    def _on_stem(self) -> None:
        if self.stem_var.get():
            self.lemma_var.set(False)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON / JSONL", "*.json *.jsonl"), ("Všechny soubory", "*.*")]
        )
        if path:
            self.custom_path_var.set(path)

    def refresh_add_combo(self) -> None:
        names = list(self.app.indexes.keys())
        self.add_target_combo["values"] = names
        if names and self.add_target_var.get() not in names:
            self.add_target_var.set(names[0])

    def _start_add(self) -> None:
        threading.Thread(target=self._do_add_documents, daemon=True).start()

    def _do_add_documents(self) -> None:
        self.app.root.after(0, lambda: self.add_btn.config(state="disabled"))
        self.app.root.after(0, lambda: self.add_status_lbl.config(text=""))

        try:
            idx_name = self.add_target_var.get()
            if not idx_name or idx_name not in self.app.indexes:
                self._log("CHYBA: vyberte cílový index."); return

            path = self.add_path_var.get().strip()
            if not path or not os.path.exists(path):
                self._log("CHYBA: zadejte platnou cestu k souboru."); return

            idx_data = self.app.indexes[idx_name]
            pre: WowheadPreprocessor = idx_data["preprocessor"]

            self._log(f"\n=== Doindexování do '{idx_name}' ===")
            self._log(f"Načítám: {path}")
            t0 = time.perf_counter()

            new_raw = load_wowhead_articles(path) if path.endswith(".jsonl") \
                else load_eval_documents(path)
            self._log(f"Načteno {len(new_raw)} nových dokumentů")

            # Existing doc ids to skip duplicates
            existing_ids = set(idx_data["doc_lookup"].keys())
            new_tuples: List[Tuple[str, List[str]]] = []
            skipped = 0
            for i, d in enumerate(new_raw):
                doc_id = str(d.get("id") or d.get("url") or f"doc_{i}")
                if doc_id in existing_ids:
                    skipped += 1
                    continue
                text = " ".join(filter(None, [
                    d.get("title", ""), d.get("content", ""), d.get("text", "")
                ]))
                tokens = pre.clean_text(text).split()
                if tokens:
                    new_tuples.append((doc_id, tokens))
                    idx_data["doc_lookup"][doc_id] = d

            if skipped:
                self._log(f"Přeskočeno {skipped} duplicitních dokumentů")
            self._log(f"Nových dokumentů k přidání: {len(new_tuples)}")

            if not new_tuples:
                self._log("Nic nového k přidání."); return

            # Update Boolean index (supports incremental add)
            idx_data["boolean"].add_documents(new_tuples)

            # Rebuild TF-IDF index with old + new tuples
            all_tuples = idx_data["doc_tuples"] + new_tuples
            new_tfidf = TfidfIndex()
            new_tfidf.build(all_tuples)
            idx_data["tfidf"] = new_tfidf
            idx_data["doc_tuples"] = all_tuples
            idx_data["lsa"] = None  # invalidate LSA cache

            elapsed = time.perf_counter() - t0
            n = new_tfidf.N
            vocab = len(new_tfidf.df)
            self._log(f"Index aktualizován: {n} dokumentů, {vocab} termů  ({elapsed:.2f}s)")
            self._log("✓ Doindexování dokončeno!\n")

            msg = f"✓ Přidáno {len(new_tuples)} dok. → celkem {n}"
            self.app.root.after(0, lambda: self.add_status_lbl.config(text=msg))
            self.app.root.after(0, self.app.tab_search.refresh_index_combo)
            self.app.root.after(0, self.app.tab_eval.refresh_index_combo)

        except Exception as exc:
            import traceback
            self._log(f"\nCHYBA: {exc}\n{traceback.format_exc()}")
        finally:
            self.app.root.after(0, lambda: self.add_btn.config(state="normal"))

    def _start_build(self) -> None:
        threading.Thread(target=self._do_build, daemon=True).start()

    def _do_build(self) -> None:
        self.app.root.after(0, lambda: self.build_btn.config(state="disabled"))
        self.app.root.after(0, lambda: self.progress.start(10))
        self.app.root.after(0, lambda: self.status_lbl.config(text=""))

        try:
            src = self.src_var.get()
            name = self.index_name_var.get().strip() or "index"

            self._log(f"=== Indexace '{name}' ===")
            t0 = time.perf_counter()

            if src == "wowhead":
                path = WOWHEAD_PATH
                if not os.path.exists(path):
                    self._log(f"CHYBA: soubor '{path}' nenalezen."); return
                self._log(f"Načítám: {path}")
                raw_docs = load_wowhead_articles(path)
            elif src == "eval":
                path = EVAL_DOCS_PATH
                if not os.path.exists(path):
                    self._log(f"CHYBA: soubor '{path}' nenalezen."); return
                self._log(f"Načítám: {path}")
                raw_docs = load_eval_documents(path)
            else:
                path = self.custom_path_var.get().strip()
                if not path or not os.path.exists(path):
                    self._log("CHYBA: zadejte platnou cestu."); return
                self._log(f"Načítám: {path}")
                raw_docs = load_wowhead_articles(path) if path.endswith(".jsonl") \
                    else load_eval_documents(path)

            self._log(f"Načteno {len(raw_docs)} dokumentů  ({time.perf_counter()-t0:.2f}s)")

            pre = WowheadPreprocessor(
                use_stemming=self.stem_var.get(),
                use_lemmatization=self.lemma_var.get(),
                remove_diacritics=self.diac_var.get(),
            )
            parts = []
            if self.lemma_var.get(): parts.append("lemmatizace")
            elif self.stem_var.get(): parts.append("stemming")
            if self.diac_var.get(): parts.append("bez diakritiky")
            self._log(f"Předzpracování: {', '.join(parts) or 'základní'} + stopwords")

            t1 = time.perf_counter()
            self._log("Tokenizace a stavba TF-IDF + boolean indexu…")
            tfidf, boolean_idx, doc_lookup, doc_tuples = build_index_from_raw(raw_docs, pre)
            idx_time = time.perf_counter() - t1
            total = time.perf_counter() - t0

            self._log(f"\nStatistiky indexu '{name}':")
            self._log(f"  Dokumenty (TF-IDF) : {tfidf.N}")
            self._log(f"  Velikost slovníku  : {len(tfidf.df)} unikátních termů")
            self._log(f"  Dokumenty (Boolean): {len(boolean_idx.all_docs)}")
            self._log(f"  Čas indexace       : {idx_time:.2f}s")
            self._log(f"  Celkový čas        : {total:.2f}s")

            self.app.register_index(name, tfidf, boolean_idx, doc_lookup, pre, raw_docs, doc_tuples)
            self._log(f"\n✓ Index '{name}' připraven!\n")

            msg = f"✓ '{name}': {tfidf.N} dok., {len(tfidf.df)} termů"
            self.app.root.after(0, lambda: self.status_lbl.config(text=msg))

        except Exception as exc:
            import traceback
            self._log(f"\nCHYBA: {exc}\n{traceback.format_exc()}")
        finally:
            self.app.root.after(0, lambda: self.progress.stop())
            self.app.root.after(0, lambda: self.build_btn.config(state="normal"))


# ---------------------------------------------------------------------------
# Tab: Vyhledávání
# ---------------------------------------------------------------------------

class SearchTab:
    def __init__(self, parent: ttk.Notebook, app: IRApp) -> None:
        self.app = app
        self.frame = ttk.Frame(parent)
        self._doc_cache: Dict[str, Dict] = {}
        self._highlight_terms: List[str] = []
        self._build_widgets()

    def _build_widgets(self) -> None:
        P = {"padx": 8, "pady": 3}

        # Query input
        q_lf = ttk.LabelFrame(self.frame, text="Dotaz")
        q_lf.pack(fill="x", **P)
        q_row = ttk.Frame(q_lf)
        q_row.pack(fill="x", padx=6, pady=6)

        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(q_row, textvariable=self.query_var,
                                     font=("Segoe UI", 11), width=62)
        self.query_entry.pack(side="left")
        self.query_entry.bind("<Return>", lambda _e: self._search())

        self.search_btn = ttk.Button(q_row, text="Hledat", command=self._search, width=9)
        self.search_btn.pack(side="left", padx=6)
        ttk.Button(q_row, text="Smazat", command=self._clear, width=7).pack(side="left")

        # Options row
        opt_row = ttk.Frame(self.frame)
        opt_row.pack(fill="x", padx=8, pady=2)

        ttk.Label(opt_row, text="Index:").pack(side="left")
        self.index_var = tk.StringVar()
        self.index_combo = ttk.Combobox(opt_row, textvariable=self.index_var,
                                        width=18, state="readonly")
        self.index_combo.pack(side="left", padx=(2, 14))

        ttk.Label(opt_row, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value="TF-IDF (vektorový)")
        self.model_combo = ttk.Combobox(
            opt_row, textvariable=self.model_var, width=26, state="readonly",
            values=["TF-IDF (vektorový)", "Boolean (AND / OR / NOT)",
                    "LSA (sémantické)", "SentenceTransformers"],
        )
        self.model_combo.pack(side="left", padx=(2, 14))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model)

        ttk.Label(opt_row, text="Top-K:").pack(side="left")
        self.topk_var = tk.IntVar(value=10)
        ttk.Spinbox(opt_row, from_=1, to=500, textvariable=self.topk_var, width=6).pack(
            side="left", padx=(2, 14))

        self.bool_hint_lbl = ttk.Label(
            opt_row,
            text="Syntaxe:  term AND term  |  term OR term  |  NOT term  |  (A OR B) AND C",
            foreground="#666", font=("Segoe UI", 8),
        )
        self.lsa_hint_lbl = ttk.Label(
            opt_row,
            text="LSA: první dotaz staví SVD (~2–5 s), pak je vyhledávání rychlé",
            foreground="#666", font=("Segoe UI", 8),
        )
        self.st_hint_lbl = ttk.Label(
            opt_row,
            text="ST: první dotaz stáhne model a zakóduje dokumenty (~10–30 s), pak rychlé",
            foreground="#666", font=("Segoe UI", 8),
        )

        # Status bar
        self.status_var = tk.StringVar(value="Vyberte index a zadejte dotaz.")
        ttk.Label(self.frame, textvariable=self.status_var,
                  foreground="#1a4e8a").pack(anchor="w", padx=8, pady=1)

        # Split pane: results list | document detail
        pane = ttk.PanedWindow(self.frame, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=4)

        list_lf = ttk.LabelFrame(pane, text="Výsledky")
        pane.add(list_lf, weight=3)

        cols = ("rank", "score", "doc_id", "title")
        self.tree = ttk.Treeview(list_lf, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("rank",  text="#",    anchor="center")
        self.tree.heading("score", text="Skóre", anchor="center")
        self.tree.heading("doc_id", text="Doc ID")
        self.tree.heading("title",  text="Název dokumentu")
        self.tree.column("rank",   width=38,  anchor="center", stretch=False)
        self.tree.column("score",  width=80,  anchor="center", stretch=False)
        self.tree.column("doc_id", width=140, stretch=False)
        self.tree.column("title",  width=360)

        vsb = ttk.Scrollbar(list_lf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        detail_lf = ttk.LabelFrame(pane, text="Náhled dokumentu")
        pane.add(detail_lf, weight=2)

        self.detail_text = scrolledtext.ScrolledText(
            detail_lf, wrap="word", font=("Segoe UI", 9), state="disabled"
        )
        self.detail_text.pack(fill="both", expand=True, padx=4, pady=4)

    def refresh_index_combo(self) -> None:
        names = list(self.app.indexes.keys())
        self.index_combo["values"] = names
        if names and self.index_var.get() not in names:
            self.index_var.set(names[0])

    def _on_model(self, _event=None) -> None:
        model = self.model_var.get()
        self.bool_hint_lbl.pack_forget()
        self.lsa_hint_lbl.pack_forget()
        self.st_hint_lbl.pack_forget()
        if "Boolean" in model:
            self.bool_hint_lbl.pack(side="left", padx=4)
        elif "LSA" in model:
            self.lsa_hint_lbl.pack(side="left", padx=4)
        elif "Sentence" in model:
            self.st_hint_lbl.pack(side="left", padx=4)

    def _clear(self) -> None:
        self.query_var.set("")
        self.tree.delete(*self.tree.get_children())
        self._doc_cache.clear()
        self.status_var.set("Dotaz vymazán.")
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.config(state="disabled")

    def _search(self) -> None:
        idx_name = self.index_var.get()
        if not idx_name or idx_name not in self.app.indexes:
            messagebox.showwarning("Vyhledávání", "Nejprve vytvořte index na záložce Indexace.")
            return
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Vyhledávání", "Zadejte dotaz.")
            return

        model = self.model_var.get()
        idx_data = self.app.indexes[idx_name]
        k = self.topk_var.get()

        self.tree.delete(*self.tree.get_children())
        self._doc_cache.clear()

        if "LSA" in model:
            self.search_btn.config(state="disabled")
            threading.Thread(
                target=self._search_lsa, args=(idx_name, query, k), daemon=True
            ).start()
            return

        if "Sentence" in model:
            self.search_btn.config(state="disabled")
            threading.Thread(
                target=self._search_st, args=(idx_name, query, k), daemon=True
            ).start()
            return

        try:
            t0 = time.perf_counter()

            if "TF-IDF" in model:
                pre: WowheadPreprocessor = idx_data["preprocessor"]
                tfidf: TfidfIndex = idx_data["tfidf"]
                tokens = pre.clean_text(query).split()
                results = tfidf.search(tokens, k=k, normalize=True)
                ms = (time.perf_counter() - t0) * 1000
                self.status_var.set(
                    f"Zobrazeno {len(results)} výsledků  |  {ms:.1f} ms  |  "
                    f"Index: {idx_name}  |  Model: TF-IDF / cosine similarity"
                )
                self._highlight_terms = [w for w in query.split() if w]
                doc_lookup = idx_data["doc_lookup"]
                for i, r in enumerate(results, 1):
                    doc = doc_lookup.get(r.doc_id, {})
                    title = (doc.get("title") or "")[:100]
                    self.tree.insert("", "end", iid=r.doc_id,
                                     values=(i, f"{r.score:.4f}", r.doc_id, title))
                    self._doc_cache[r.doc_id] = doc

            else:  # Boolean
                pre: WowheadPreprocessor = idx_data["preprocessor"]
                bidx: BooleanIndex = idx_data["boolean"]
                processed_q = preprocess_boolean_query_terms(query, pre)
                hits = bidx.search(processed_q)
                total = len(hits)
                ms = (time.perf_counter() - t0) * 1000
                self.status_var.set(
                    f"Nalezeno {total} dokumentů  |  Zobrazeno: {min(k, total)}  |  "
                    f"{ms:.1f} ms  |  Index: {idx_name}  |  Model: Boolean"
                )
                self._highlight_terms = [
                    t for t in tokenize_boolean_query(query)
                    if t not in {"AND", "OR", "NOT", "(", ")"}
                ]
                doc_lookup = idx_data["doc_lookup"]
                for i, doc_id in enumerate(hits[:k], 1):
                    doc = doc_lookup.get(doc_id, {})
                    title = (doc.get("title") or "")[:100]
                    self.tree.insert("", "end", iid=doc_id,
                                     values=(i, "—", doc_id, title))
                    self._doc_cache[doc_id] = doc

        except BooleanQueryParseError as e:
            messagebox.showerror(
                "Chybný boolean dotaz",
                f"Dotaz nelze zpracovat:\n{e}\n\n"
                "Příklady:\n"
                "  warcraft AND shadowlands\n"
                "  (dragon OR raid) AND NOT pvp\n"
                "  NOT arena AND battleground",
            )
        except Exception as e:
            import traceback
            messagebox.showerror("Chyba vyhledávání", f"{e}\n\n{traceback.format_exc()[:600]}")

    def _search_lsa(self, idx_name: str, query: str, k: int) -> None:
        """LSA search runs in a background thread; builds SVD index on first use."""
        try:
            idx_data = self.app.indexes[idx_name]

            # Lazy-build LSA index and cache it inside the index data dict
            if idx_data.get("lsa") is None:
                self.app.root.after(0, lambda: self.status_var.set(
                    "Stavím LSA index (SVD decomposition) — chvíli strpení…"
                ))
                lsa = LsaIndex(k=200)
                lsa.build(idx_data["doc_tuples"])
                idx_data["lsa"] = lsa

            lsa: LsaIndex = idx_data["lsa"]

            t0 = time.perf_counter()
            results = lsa.search(query, top_k=k)
            ms = (time.perf_counter() - t0) * 1000

            doc_lookup = idx_data["doc_lookup"]
            rows = []
            cache: Dict[str, Dict] = {}
            for i, (doc_id, score) in enumerate(results, 1):
                doc = doc_lookup.get(doc_id, {})
                title = (doc.get("title") or "")[:100]
                rows.append((i, f"{score:.4f}", doc_id, title))
                cache[doc_id] = doc

            status = (
                f"Zobrazeno {len(results)} výsledků  |  {ms:.1f} ms  |  "
                f"Index: {idx_name}  |  Model: LSA / cosine similarity (latentní prostor)"
            )

            terms = [w for w in query.split() if w]

            def update_ui() -> None:
                self.tree.delete(*self.tree.get_children())
                self._doc_cache.clear()
                self._doc_cache.update(cache)
                self._highlight_terms = terms
                for row in rows:
                    self.tree.insert("", "end", iid=row[2], values=row)
                self.status_var.set(status)

            self.app.root.after(0, update_ui)

        except Exception as e:
            import traceback
            err = str(e)
            tb = traceback.format_exc()[:500]
            self.app.root.after(
                0, lambda: messagebox.showerror("Chyba LSA vyhledávání", f"{err}\n\n{tb}")
            )
        finally:
            self.app.root.after(0, lambda: self.search_btn.config(state="normal"))

    def _search_st(self, idx_name: str, query: str, k: int) -> None:
        """SentenceTransformers search — lazy-builds and caches document embeddings."""
        try:
            import numpy as np
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self.app.root.after(0, lambda: messagebox.showerror(
                    "Chybí balíček", "Nainstaluj:\n  pip install sentence-transformers"
                ))
                return

            idx_data = self.app.indexes[idx_name]

            if idx_data.get("st") is None:
                self.app.root.after(0, lambda: self.status_var.set(
                    "Načítám SentenceTransformers model a kóduji dokumenty… (první dotaz)"
                ))
                raw_docs: List[Dict] = idx_data["raw_docs"]
                doc_ids: List[str] = []
                doc_texts: List[str] = []
                for i, d in enumerate(raw_docs):
                    doc_ids.append(str(d.get("id") or d.get("url") or f"doc_{i}"))
                    doc_texts.append(" ".join(filter(None, [
                        d.get("title", ""), d.get("content", ""), d.get("text", "")
                    ])))
                model = SentenceTransformer("all-MiniLM-L6-v2")
                doc_emb = model.encode(doc_texts, batch_size=64,
                                       normalize_embeddings=True, show_progress_bar=False)
                idx_data["st"] = {"model": model, "doc_emb": doc_emb, "doc_ids": doc_ids}

            st = idx_data["st"]
            model = st["model"]
            doc_emb = st["doc_emb"]
            doc_ids = st["doc_ids"]

            t0 = time.perf_counter()
            q_emb = model.encode([query], normalize_embeddings=True)
            scores = doc_emb @ q_emb[0]
            best = np.argsort(scores)[::-1][:k]
            ms = (time.perf_counter() - t0) * 1000

            doc_lookup = idx_data["doc_lookup"]
            rows = []
            cache: Dict[str, Dict] = {}
            for rank, i in enumerate(best, 1):
                doc_id = doc_ids[i]
                doc = doc_lookup.get(doc_id, {})
                title = (doc.get("title") or "")[:100]
                rows.append((rank, f"{scores[i]:.4f}", doc_id, title))
                cache[doc_id] = doc

            terms = [w for w in query.split() if w]
            status = (
                f"Zobrazeno {len(rows)} výsledků  |  {ms:.1f} ms  |  "
                f"Index: {idx_name}  |  Model: SentenceTransformers (all-MiniLM-L6-v2)"
            )

            def update_ui() -> None:
                self.tree.delete(*self.tree.get_children())
                self._doc_cache.clear()
                self._doc_cache.update(cache)
                self._highlight_terms = terms
                for row in rows:
                    self.tree.insert("", "end", iid=row[2], values=row)
                self.status_var.set(status)

            self.app.root.after(0, update_ui)

        except Exception as e:
            import traceback
            err = str(e)
            tb = traceback.format_exc()[:500]
            self.app.root.after(
                0, lambda: messagebox.showerror("Chyba ST vyhledávání", f"{err}\n\n{tb}")
            )
        finally:
            self.app.root.after(0, lambda: self.search_btn.config(state="normal"))

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        doc_id = sel[0]
        doc = self._doc_cache.get(doc_id, {})

        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")

        self.detail_text.insert("end", f"ID:    {doc_id}\n")
        for field, label in [("title", "Název"), ("url", "URL"),
                              ("author", "Autor"), ("date", "Datum")]:
            val = doc.get(field)
            if val:
                self.detail_text.insert("end", f"{label}:  {val}\n")

        self.detail_text.insert("end", "\n" + "─" * 60 + "\n\n")
        content = doc.get("content") or doc.get("text") or ""
        self.detail_text.insert("end", content[:5000])
        if len(content) > 5000:
            self.detail_text.insert("end", "\n\n[… text zkrácen …]")

        self._apply_highlights()
        self.detail_text.config(state="disabled")

    def _apply_highlights(self) -> None:
        self.detail_text.tag_config("hl", background="#FFE066", foreground="#000")
        self.detail_text.tag_remove("hl", "1.0", "end")
        for term in self._highlight_terms:
            if not term:
                continue
            start = "1.0"
            while True:
                pos = self.detail_text.search(term, start, stopindex="end", nocase=True)
                if not pos:
                    break
                end_pos = f"{pos}+{len(term)}c"
                self.detail_text.tag_add("hl", pos, end_pos)
                start = end_pos


# ---------------------------------------------------------------------------
# Tab: Evaluace
# ---------------------------------------------------------------------------

class EvalTab(LogMixin):
    def __init__(self, parent: ttk.Notebook, app: IRApp) -> None:
        self.app = app
        self.frame = ttk.Frame(parent)
        self._last_run: Optional[Dict[str, List[str]]] = None
        self._last_run_scores: Optional[Dict[str, List[Tuple[str, float]]]] = None
        self._last_queries: Optional[List[QueryItem]] = None
        self._build_widgets()

    def _build_widgets(self) -> None:
        P = {"padx": 8, "pady": 4}

        opt_lf = ttk.LabelFrame(self.frame, text="Nastavení evaluace")
        opt_lf.pack(fill="x", **P)

        r1 = ttk.Frame(opt_lf)
        r1.pack(fill="x", padx=6, pady=4)
        ttk.Label(r1, text="Index:").pack(side="left")
        self.eval_index_var = tk.StringVar()
        self.eval_index_combo = ttk.Combobox(r1, textvariable=self.eval_index_var,
                                             width=20, state="readonly")
        self.eval_index_combo.pack(side="left", padx=(2, 16))
        ttk.Label(r1, text="Soubor dotazů:").pack(side="left")
        self.queries_path_var = tk.StringVar(value=EVAL_QUERIES_PATH)
        ttk.Entry(r1, textvariable=self.queries_path_var, width=38).pack(side="left", padx=(2, 2))
        ttk.Button(r1, text="…", command=self._browse_queries, width=3).pack(side="left")

        r2 = ttk.Frame(opt_lf)
        r2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(r2, text="Top-K:").pack(side="left")
        self.topk_var = tk.IntVar(value=50)
        ttk.Spinbox(r2, from_=5, to=500, textvariable=self.topk_var, width=6).pack(
            side="left", padx=(2, 16))
        ttk.Label(r2, text="Metoda:").pack(side="left")
        self.method_var = tk.StringVar(value="TF-IDF")
        ttk.Combobox(r2, textvariable=self.method_var, width=22, state="readonly",
                     values=["TF-IDF", "LSA", "SentenceTransformers"]).pack(side="left", padx=(2, 0))

        btn_row = ttk.Frame(self.frame)
        btn_row.pack(fill="x", **P)
        self.run_btn = ttk.Button(btn_row, text="▶  Spustit evaluaci",
                                  command=self._start_eval, width=20)
        self.run_btn.pack(side="left")
        self.export_btn = ttk.Button(btn_row, text="Exportovat výsledky (TREC)",
                                     command=self._export_trec, width=26)
        self.export_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(btn_row, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=4)

        out_lf = ttk.LabelFrame(self.frame, text="Výsledky evaluace")
        out_lf.pack(fill="both", expand=True, **P)
        log_widget = scrolledtext.ScrolledText(out_lf, font=("Consolas", 9), state="disabled")
        log_widget.pack(fill="both", expand=True, padx=4, pady=4)
        self._init_log(log_widget, self.app.root)

    def refresh_index_combo(self) -> None:
        names = list(self.app.indexes.keys())
        self.eval_index_combo["values"] = names
        if names and self.eval_index_var.get() not in names:
            self.eval_index_var.set(names[0])

    def _browse_queries(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("Všechny", "*.*")]
        )
        if path:
            self.queries_path_var.set(path)

    def _start_eval(self) -> None:
        idx_name = self.eval_index_var.get()
        if not idx_name or idx_name not in self.app.indexes:
            messagebox.showwarning("Evaluace", "Vyberte index pro evaluaci.")
            return
        self._clear_log_widget()
        threading.Thread(target=self._do_eval, daemon=True).start()

    def _do_eval(self) -> None:
        self.app.root.after(0, lambda: self.run_btn.config(state="disabled"))
        self.app.root.after(0, lambda: self.progress.start(10))

        try:
            idx_name = self.eval_index_var.get()
            idx_data = self.app.indexes[idx_name]
            queries_path = self.queries_path_var.get().strip()
            k = self.topk_var.get()
            method = self.method_var.get()

            self._log(f"=== Evaluace '{idx_name}' — metoda: {method}, top-{k} ===\n")

            if not os.path.exists(queries_path):
                self._log(f"CHYBA: soubor dotazů '{queries_path}' nenalezen."); return

            self._log(f"Načítám dotazy: {queries_path}")
            queries = load_eval_queries(queries_path)
            self._last_queries = queries
            self._log(f"Počet dotazů: {len(queries)}\n")

            qrels = {q.qid: q.relevant_doc_ids for q in queries}
            run: Dict[str, List[str]] = {}
            run_scores: Dict[str, List[Tuple[str, float]]] = {}

            t0 = time.perf_counter()

            if method == "TF-IDF":
                tfidf: TfidfIndex = idx_data["tfidf"]
                pre: WowheadPreprocessor = idx_data["preprocessor"]
                self._log("Vyhledávám TF-IDF přes všechny dotazy…")
                for q in queries:
                    tokens = pre.clean_text(q.text).split()
                    results = tfidf.search(tokens, k=k, normalize=True)
                    run[q.qid] = [r.doc_id for r in results]
                    run_scores[q.qid] = [(r.doc_id, r.score) for r in results]

            elif method == "LSA":
                self._log("Stavím LSA index (SVD decomposition)…")
                raw_docs: List[Dict] = idx_data["raw_docs"]
                pre_lsa = WowheadPreprocessor(use_lemmatization=True, remove_diacritics=True)
                lsa_tuples: List[Tuple[str, List[str]]] = []
                for i, d in enumerate(raw_docs):
                    doc_id = str(d.get("id") or d.get("url") or f"doc_{i}")
                    text = " ".join(filter(None, [
                        d.get("title", ""), d.get("content", ""), d.get("text", "")
                    ]))
                    tokens = pre_lsa.clean_text(text).split()
                    if tokens:
                        lsa_tuples.append((doc_id, tokens))
                lsa = LsaIndex(k=200)
                lsa.build(lsa_tuples)
                self._log(f"LSA index postaven ({len(lsa_tuples)} dokumentů). Vyhledávám…")
                for q in queries:
                    results = lsa.search(q.text, top_k=k)
                    run[q.qid] = [did for did, _ in results]
                    run_scores[q.qid] = results

            else:  # SentenceTransformers
                self._log("Načítám SentenceTransformers model (all-MiniLM-L6-v2)…")
                self._log("(První spuštění může stáhnout model ~80 MB)\n")
                try:
                    from sentence_transformers import SentenceTransformer
                    import numpy as np
                except ImportError:
                    self._log("CHYBA: pip install sentence-transformers"); return

                raw_docs = idx_data["raw_docs"]
                doc_ids: List[str] = []
                doc_texts: List[str] = []
                for i, d in enumerate(raw_docs):
                    doc_ids.append(str(d.get("id") or d.get("url") or f"doc_{i}"))
                    doc_texts.append(" ".join(filter(None, [
                        d.get("title", ""), d.get("content", ""), d.get("text", "")
                    ])))

                model = SentenceTransformer("all-MiniLM-L6-v2")
                self._log(f"Kóduji {len(doc_texts)} dokumentů…")
                doc_emb = model.encode(doc_texts, batch_size=64, normalize_embeddings=True,
                                       show_progress_bar=False)
                self._log("Vyhledávám…")
                for q in queries:
                    q_emb = model.encode([q.text], normalize_embeddings=True)
                    scores = doc_emb @ q_emb[0]
                    best = np.argsort(scores)[::-1][:k]
                    run[q.qid] = [doc_ids[i] for i in best]
                    run_scores[q.qid] = [(doc_ids[i], float(scores[i])) for i in best]

            elapsed = time.perf_counter() - t0
            self._last_run = run
            self._last_run_scores = run_scores

            self._log(f"\nCelkový čas vyhledávání : {elapsed:.2f}s")
            self._log(f"Průměr na dotaz          : {elapsed / max(len(queries), 1) * 1000:.1f} ms")
            self._log("\n" + "─" * 44)
            for cutoff in [10, 20, 50]:
                val = mean_average_precision(run, qrels, k=cutoff)
                self._log(f"  MAP@{cutoff:<3d} : {val:.4f}")
            self._log("─" * 44)
            self._log("\n✓ Evaluace dokončena.")

        except Exception as exc:
            import traceback
            self._log(f"\nCHYBA: {exc}\n{traceback.format_exc()}")
        finally:
            self.app.root.after(0, lambda: self.progress.stop())
            self.app.root.after(0, lambda: self.run_btn.config(state="normal"))

    def _export_trec(self) -> None:
        if not self._last_run_scores:
            messagebox.showwarning("Export", "Nejprve spusťte evaluaci.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Všechny", "*.*")],
            initialfile="results.txt",
        )
        if not path:
            return
        run_id = f"{self.method_var.get().lower()}_run"
        with open(path, "w", encoding="utf-8") as f:
            for qid, scored in sorted(self._last_run_scores.items()):
                for rank, (doc_id, score) in enumerate(scored, 1):
                    f.write(f"{qid} 0 {doc_id} {rank} {score:.6f} {run_id}\n")
        messagebox.showinfo("Export dokončen", f"Výsledky uloženy:\n{path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    IRApp(root)
    root.mainloop()
