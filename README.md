================================================================================
  IR SYSTÉM — Automatická indexace a vyhledávání dokumentů
  Semestrální práce, KIV/IR
================================================================================

POŽADAVKY
---------
  Python 3.10 nebo novější

  Závislosti (instalace viz níže):
    nltk, numpy, scipy, pandas, unidecode, num2words, selenium, sentence-transformers

INSTALACE ZÁVISLOSTÍ
--------------------
  pip install -r requirements.txt

  Po instalaci je nutné stáhnout NLTK data (automaticky při prvním spuštění):
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

STRUKTURA PROJEKTU
------------------
  gui.py                  — hlavní spustitelný soubor (grafické rozhraní)   
  preprocessing.py        — tokenizace a předzpracování textu   
  tfidf_search.py         — TF-IDF index a cosine similarity    
  boolean_search.py       — Boolean index a parser dotazů   
  eval_ir.py              — evaluační framework (MAP, LSA, SentenceTransformers)    
  run_tfidf.py            — CLI demo pro TF-IDF vyhledávání 
  run_boolean.py          — CLI demo pro Boolean vyhledávání    
  run_semantics.py        — CLI demo pro LSA vyhledávání    
  crawler.py              — webový crawler pro sběr Wowhead článků  
  requirements.txt        — seznam Python závislostí

UMÍSTĚNÍ DAT
------------
  Wowhead articles:     
  * wowhead_articles.jsonl        ← umístit do kořenového adresáře projektu

  Evaluační data:   
  * documents.json           ← umístit do podadresáře data/
  * full_text_queries.json   ← umístit do podadresáře data/
  * gold_relevancies.txt     ← umístit do podadresáře data/

  Formát wowhead_articles.jsonl (jeden JSON objekt na řádek):   
    {"title": "...", "author": "...", "date": "...", "content": "...", "url": "..."}

  Formát data/documents.json (JSON pole):   
    [{"id": "d0", "title": "...", "text": "..."}, ...]

  Vlastní data lze přidat ve stejném formátu (JSONL = wowhead formát, JSON pole = evaluační formát).

SPUŠTĚNÍ — GRAFICKÉ ROZHRANÍ (GUI)
------------------------------------
  python gui.py

  Program otevře okno se třemi záložkami:
  1. Indexace      — načtení a zaindexování dat
  2. Vyhledávání   — zadávání dotazů a prohlížení výsledků
  3. Evaluace      — výpočet MAP metriky nad evaluačními daty

RYCHLÝ PRŮVODCE GUI
--------------------
  1. INDEXACE:
     - Vyberte zdroj dat (Wowhead / Evaluační / Vlastní soubor)    
     - Zvolte název indexu (lze mít více indexů najednou)  
     - Nastavte předzpracování (lemmatizace doporučena)    
     - Klikněte "Vytvořit index"   
     - Volitelně: doindexujte další data v sekci "Doindexovat"

  2. VYHLEDÁVÁNÍ:  
     - Vyberte index a model vyhledávání:
        - TF-IDF (vektorový)      — přirozený jazyk, vrací top-K dle relevance
        - Boolean (AND / OR / NOT) — logické operátory, závorky
        - LSA (sémantické)         — latentní sémantická analýza  
     - Zadejte dotaz (Enter nebo tlačítko Hledat)  
     - Kliknutím na výsledek zobrazíte náhled dokumentu  
        (hledaná slova jsou zvýrazněna žlutě)

  3. EVALUACE:
     - Vyberte index (doporučeno: evaluační data)
     - Ověřte cestu k souboru dotazů
     - Klikněte "Spustit evaluaci"
     - Výsledky MAP@10, MAP@20, MAP@50 se zobrazí v logu
     - Výsledky lze exportovat ve TREC formátu tlačítkem "Exportovat výsledky"

SYNTAXE BOOLEAN DOTAZŮ
-----------------------
  Operátory (case-insensitive):     
  AND, OR, NOT    
  Závorky pro prioritu: ( )     
  Priorita: NOT > AND > OR

  Příklady:     
    warcraft AND shadowlands    
    (dragon OR raid) AND NOT pvp    
    NOT arena AND battleground OR guild

SPUŠTĚNÍ CRAWLERU (volitelné)
------------------------------
  Crawler vyžaduje Firefox a geckodriver v PATH.
    python crawler.py

  Výstup se uloží do wowhead_articles.jsonl

POZNÁMKY
--------
  - LSA vyhledávání staví SVD index při prvním dotazu (~2-5 sekund).
    Každý další dotaz je okamžitý (index je cachován v paměti).
  - SentenceTransformers model (all-MiniLM-L6-v2) se stáhne automaticky
    při prvním použití v evaluaci (~80 MB).
  - Indexace 711 Wowhead článků trvá přibližně 2-5 sekund.
  - Indexace 400 evaluačních dokumentů trvá přibližně 1-2 sekundy.
  - Všechny indexy jsou in-memory; zavřením GUI se zahodí.

================================================================================