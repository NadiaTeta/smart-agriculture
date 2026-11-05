"""
Umurima: Smart Farming Assistant (Streamlit version)

Retrieval-Augmented Generation (RAG):
  1. The farmer's question is turned into a vector (BAAI/bge-small-en-v1.5).
  2. We find the 4 most similar chunks from 23 trusted farming guides (cosine similarity).
  3. Those chunks + the question are sent to an LLM on Groq, which answers only from them
     and cites the passages it used.
"""
import base64
import html
import json
import os
import pathlib
import re

import numpy as np
import streamlit as st

ROOT = pathlib.Path(__file__).parent
APP_NAME = "Umurima"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
REFUSAL = "don't have enough information"
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]   # Groq models (first one is tried first)

st.set_page_config(page_title=f"{APP_NAME} · Smart Farming Assistant", page_icon="🌱",
                   layout="wide", initial_sidebar_state="collapsed")

CROPS = [  # (label, crop key used in the index)
    ("🌱 All crops", None), ("🌽 Maize", "maize"), ("🫘 Beans", "beans"),
    ("🥔 Potato", "potato"), ("🌿 Cassava", "cassava"), ("🍌 Banana", "banana"),
]
SUGGESTIONS = {
    None: ["How can I control fall armyworm on my maize?",
           "My bean roots look brown and rotten. What should I do?",
           "How do I recognise late blight on potatoes?",
           "How do I stop banana wilt from spreading?"],
    "maize": ["How can I control fall armyworm on my maize?",
              "What spacing should I use when planting maize?",
              "When should I apply fertilizer to maize?",
              "What is site-specific fertiliser management for maize?"],
    "beans": ["Which climbing bean varieties does RAB recommend?",
              "How do I control bean fly?",
              "My bean roots look brown and rotten. What should I do?",
              "What is rhizobium inoculation for beans?"],
    "potato": ["How do I recognise late blight on potatoes?",
               "How should I store seed potatoes until next season?",
               "How do I choose good seed potatoes?",
               "Which fungicides are used against late blight?"],
    "cassava": ["How do I select healthy cassava cuttings?",
                "What are the symptoms of cassava brown streak disease?",
                "How is cassava mosaic disease spread?",
                "How do I choose a good site for planting cassava?"],
    "banana": ["What are the symptoms of banana Xanthomonas wilt?",
               "How do I stop banana wilt from spreading?",
               "Why should I remove the male bud of banana plants?",
               "How do I clean my tools to avoid spreading disease?"],
}


# ----------------------------------------------------------------- loading (cached)
@st.cache_resource(show_spinner="Loading the farming guides…")
def load_knowledge():
    chunks = json.loads((ROOT / "data/chunks.json").read_text())
    emb = np.load(ROOT / "data/embeddings.npy").astype("float32")
    docs = json.loads((ROOT / "data/documents.json").read_text())
    prompt = (ROOT / "data/system_prompt.txt").read_text()
    return chunks, emb, docs, prompt


@st.cache_resource(show_spinner="Loading the search model…")
def load_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding("BAAI/bge-small-en-v1.5")


@st.cache_data
def photo(name):
    p = ROOT / "assets" / f"{name}.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


CHUNKS, EMB, DOCS, SYSTEM_PROMPT = load_knowledge()
CREDITS = json.loads((ROOT / "assets/credits.json").read_text())
N_DOCS = len(DOCS)


# ----------------------------------------------------------------- RAG
def retrieve(question, crop=None, k=4):
    q = np.array(next(iter(load_embedder().embed([QUERY_PREFIX + question]))), dtype="float32")
    q /= np.linalg.norm(q)
    scores = EMB @ q                       # cosine similarity (all vectors are normalised)
    results = []
    for i in np.argsort(-scores):
        if crop and CHUNKS[i]["crop"] != crop:
            continue
        results.append({**CHUNKS[i], "score": float(scores[i])})
        if len(results) == k:
            break
    return results


def build_messages(question, results):
    context = "\n\n".join(f"[{i + 1}] (Source: {r['source']}, page {r['page']})\n{r['text']}"
                          for i, r in enumerate(results))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:  # no secrets file or key not set
        return os.environ.get(name)


def get_api_key():
    return get_secret("GROQ_API_KEY")


def stream_answer(messages):
    """Yield the answer piece by piece from Groq. Tries the next model if one fails before writing."""
    if os.environ.get("UMURIMA_FAKE_LLM"):            # offline testing only
        import time
        text = ("Here is what the guides say [1]:\n\n- Check your plants regularly for early signs.\n"
                "- Remove and destroy infected plants [2].\n- Clean your tools after every plant [1].")
        for word in text.split(" "):
            time.sleep(0.03)
            yield word + " "
        return

    from groq import Groq
    client = Groq(api_key=get_api_key())
    models = [m for m in [get_secret("GROQ_MODEL")] if m] + MODELS   # optional override in Secrets
    last_error = None
    attempts = [(m, extra) for m in dict.fromkeys(models) for extra in ({"reasoning_effort": "low"}, {})]
    for model, extra in attempts:
        wrote = False
        try:
            stream = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, stream=True,
                max_completion_tokens=1500, **extra)
            for part in stream:
                delta = part.choices[0].delta.content
                if delta:
                    wrote = True
                    yield delta
            return
        except Exception as e:  # noqa: BLE001
            last_error = e
            if wrote:
                raise
    raise last_error


def clean_answer(text):
    lines = text.strip().split("\n")
    only_cites = re.compile(r"^[\W_]*(sources?|citations?|references?)?[\s:]*(\[\d+\][\s,;and]*)+[\W_]*$", re.I)
    while lines and (only_cites.match(lines[-1].strip()) or not lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


# ----------------------------------------------------------------- source cards
RW_FLAG = ('<svg class="flag" viewBox="0 0 30 20" aria-hidden="true"><rect width="30" height="10" fill="#20603d"/>'
           '<rect width="30" height="20" fill="#20603d"/><rect width="30" height="13.3" fill="#fad201"/>'
           '<rect width="30" height="10" fill="#00a1de"/><circle cx="24" cy="4.2" r="2.3" fill="#e5be01"/></svg>')

def is_rwandan(source):
    return "Rwanda" in source or "RAB" in source


def sources_html(answer, results):
    if REFUSAL in answer.lower():
        return ('<div class="src-empty"><div class="src-empty-icon">🔎</div><b>No matching guide found</b>'
                '<p>This question is outside the farming guides. Try asking about maize, beans, '
                'potatoes, cassava or bananas.</p></div>')
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(results)})
    cited = cited or list(range(1, len(results) + 1))
    cards = []
    for n in cited:
        r = results[n - 1]
        url = DOCS.get(r["file"], {}).get("url", "")
        link = (f'<a class="src-link" href="{html.escape(url)}#page={r["page"]}" target="_blank" '
                f'rel="noopener">Open guide ↗</a>') if url else ""
        badge = f'<span class="badge rw">{RW_FLAG} Rwanda guide</span>' if is_rwandan(r["source"]) else ""
        excerpt = html.escape(r["text"][:600]) + ("…" if len(r["text"]) > 600 else "")
        cards.append(
            f'<div class="src-card"><div class="src-top"><span class="src-num">{n}</span>'
            f'<span class="badge crop">{html.escape(r["crop"].title())}</span>{badge}</div>'
            f'<div class="src-title">{html.escape(r["source"])}</div>'
            f'<div class="src-meta">Page {r["page"]} · match {r["score"]:.0%} {link}</div>'
            f'<details><summary>Read the passage</summary><p>{excerpt}</p></details></div>')
    return '<div class="src-head">📚 Where this answer comes from</div>' + "".join(cards)


SOURCES_IDLE = ('<div class="src-empty"><div class="src-empty-icon">📚</div><b>Sources appear here</b>'
                '<p>Every answer shows the farming guide and page it came from, so you can check the advice yourself.</p></div>')
SOURCES_SEARCHING = ('<div class="src-empty"><div class="src-empty-icon spin">🌱</div><b>Searching the farming guides…</b>'
                     '<p>Finding the passages that answer your question.</p></div>')


# ----------------------------------------------------------------- styling
LOGO_SVG = """<svg viewBox="0 0 48 48" class="logo" aria-hidden="true"><circle cx="24" cy="24" r="23" fill="#e8f5e2"/>
<path d="M24 38 C 24 28 24 22 24 14" stroke="#2e7d32" stroke-width="3" stroke-linecap="round"/>
<path d="M24 26 C 14 26 10 18 11 12 C 19 12 24 18 24 26Z" fill="#43a047"/>
<path d="M24 22 C 32 22 37 15 36 9 C 28 9 24 15 24 22Z" fill="#66bb6a"/>
<path d="M12 38 Q 24 33 36 38" stroke="#8d5a2b" stroke-width="3" fill="none" stroke-linecap="round"/></svg>"""


def css():
    tiles = []
    for label, crop in CROPS:
        key = crop or "all"
        img = (f"linear-gradient(180deg, rgba(0,0,0,0) 30%, rgba(10,30,15,.85) 100%), url('{photo(crop)}')"
               if crop and photo(crop) else "radial-gradient(circle at 30% 20%, #8bc34a, #2e7d32 60%, #173d24)")
        tiles.append(f".st-key-crop_{key} button {{ background-image: {img} !important; }}")
    selected = st.session_state.get("crop") or "all"
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
:root { --green-900:#173d24; --green-700:#2e7d32; --green-600:#3f8f4a; --green-100:#e8f5e2;
        --cream:#fbf8ef; --soil:#8d5a2b; --sun:#f6c247; --ink:#1f2a24; --muted:#5d6b62; }
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--cream) !important; }
.stApp, .stApp * { font-family: 'Nunito', system-ui, sans-serif; }
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
header[data-testid="stHeader"], #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { display:none !important; }
.block-container { max-width: 1180px !important; padding: 1rem 1rem 2rem !important; }
@keyframes rise { from { opacity:0; transform: translateY(14px);} to {opacity:1; transform:none;} }

/* Header */
.topbar { display:flex; justify-content:space-between; align-items:center; padding:6px 4px 12px; }
.brand { display:flex; gap:12px; align-items:center; } .logo { width:46px; height:46px; }
.brand-name { font-size:24px; font-weight:900; color:var(--green-900); letter-spacing:-.5px; line-height:1; }
.brand-sub { font-size:13px; color:var(--muted); font-weight:700; }
.status { background:#fff; border:1px solid #dfe8d8; padding:7px 14px; border-radius:999px; font-size:13px;
          font-weight:700; color:var(--green-900); display:flex; align-items:center; gap:8px; }
.dot { width:9px; height:9px; border-radius:50%; background:#3fbf5a; animation: pulse 2s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(63,191,90,.55)} 70%{box-shadow:0 0 0 9px rgba(63,191,90,0)} 100%{box-shadow:0 0 0 0 rgba(63,191,90,0)} }

/* Hero */
.hero { position:relative; overflow:hidden; border-radius:30px; min-height:400px; display:flex; align-items:center;
        background: linear-gradient(135deg,#2e7d32,#173d24); box-shadow: 0 18px 40px -22px rgba(23,61,36,.6);
        animation: rise .7s ease-out both; }
.hero-bg { position:absolute; inset:-2%; background-size:cover; background-position:center 45%;
           animation: kenburns 30s ease-in-out infinite alternate; }
@keyframes kenburns { from { transform: scale(1.02); } to { transform: scale(1.15) translate(-2.5%, -2%); } }
.hero-shade { position:absolute; inset:0; background: linear-gradient(90deg, rgba(12,36,20,.90) 0%, rgba(12,36,20,.70) 42%, rgba(12,36,20,.10) 78%, rgba(12,36,20,0) 100%); }
.hero-content { position:relative; z-index:1; padding:44px; max-width:620px; }
.hero-content > * { animation: rise .8s ease-out both; }
.hero-content h1 { animation-delay:.1s } .hero-content p { animation-delay:.2s } .hero-content .stats { animation-delay:.3s }
.pill { display:inline-block; background:rgba(255,255,255,.16); color:#fff; font-weight:800; font-size:13px;
        padding:6px 12px; border-radius:999px; border:1px solid rgba(255,255,255,.3); backdrop-filter: blur(6px); }
.hero h1 { font-size:42px !important; line-height:1.1 !important; margin:16px 0 12px !important; color:#fff !important;
           font-weight:900 !important; letter-spacing:-1px; padding:0 !important; }
.hl { color:#ffd76a; }
.flag { width:18px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:4px; display:inline-block; }
.hero p { color:rgba(255,255,255,.9) !important; font-size:16.5px; line-height:1.55; margin:0; }
.stats { display:flex; gap:12px; margin-top:22px; flex-wrap:wrap; }
.stats div { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28); backdrop-filter: blur(8px);
             border-radius:16px; padding:10px 16px; min-width:110px; }
.stats b { display:block; font-size:24px; color:#ffe08a; font-weight:900; }
.stats span { font-size:12.5px; color:rgba(255,255,255,.88); font-weight:700; }
.photo-tag { position:absolute; right:16px; bottom:14px; z-index:1; background:rgba(0,0,0,.38); color:#fff; font-size:12px;
             font-weight:700; padding:5px 11px; border-radius:999px; backdrop-filter: blur(6px); }

.section-title { font-weight:900; color:var(--green-900); font-size:15px; margin: 22px 2px 8px; }

/* Crop tiles (buttons with photos) */
[class*="st-key-crop_"] button { height:104px; width:100%; border-radius:18px !important; border:3px solid transparent !important;
    background-color:#2e7d32 !important; background-size:cover !important; background-position:center !important;
    display:flex !important; align-items:flex-end !important; justify-content:flex-start !important; padding:0 12px 10px !important;
    box-shadow: 0 8px 18px -12px rgba(0,0,0,.55); transition: transform .2s ease, box-shadow .2s ease; }
[class*="st-key-crop_"] button p { color:#fff !important; font-weight:900 !important; font-size:15px !important;
    text-shadow: 0 1px 6px rgba(0,0,0,.6); }
[class*="st-key-crop_"] button:hover { transform: translateY(-4px); box-shadow: 0 16px 26px -14px rgba(0,0,0,.6); }
.st-key-crop_SELECTED button { border-color: var(--sun) !important; box-shadow: 0 0 0 4px rgba(246,194,71,.35), 0 14px 26px -14px rgba(0,0,0,.6) !important; position:relative; }
.st-key-crop_SELECTED button::after { content:"✓"; position:absolute; top:8px; right:8px; width:24px; height:24px; border-radius:50%;
    background:var(--sun); color:#173d24; font-weight:900; display:flex; align-items:center; justify-content:center; font-size:14px; }
TILES

/* Chat card */
.st-key-chatcard { background:#fff; border:1px solid #e3ecd9; border-radius:26px; padding:14px 16px;
                   box-shadow: 0 12px 30px -20px rgba(23,61,36,.4); }
.st-key-chatbox { background:#fcfdf9; border-radius:18px; border:none !important; }
[data-testid="stChatMessage"] { background: transparent !important; animation: rise .35s ease-out both; }
[data-testid="stChatMessage"]:has(.u-msg) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has(.u-msg) [data-testid="stChatMessageContent"] { flex: 0 1 auto; max-width: 80%;
    background: var(--green-700); color:#fff; border-radius:18px 18px 4px 18px; padding: 10px 14px; }
[data-testid="stChatMessage"]:has(.u-msg) [data-testid="stChatMessageContent"] * { color:#fff !important; }
[data-testid="stChatMessage"]:has(.a-msg) [data-testid="stChatMessageContent"] {
    background:#f3f8ee; border:1px solid #e1ecd6; border-radius:18px 18px 18px 4px; padding: 10px 14px; color: var(--ink); }
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li { font-size:15.5px; line-height:1.6; }
.welcome { text-align:center; padding: 40px 18px; color: var(--muted); }
.welcome h3 { color: var(--green-900); font-size:22px; margin: 8px 0; font-weight:900; }
.welcome-emoji { font-size:42px; animation: wave 2.2s ease-in-out infinite; display:inline-block; transform-origin: 70% 70%; }
@keyframes wave { 0%,60%,100% { rotate:0deg } 10%,30% { rotate:14deg } 20% { rotate:-8deg } 40% { rotate:-4deg } 50% { rotate:10deg } }
.suggest-title { font-weight:900; color:var(--green-900); margin: 10px 2px 4px; font-size:14px; }
[class*="st-key-sugg_"] button { background:#fff !important; border:1.5px dashed #cfe0c3 !important; border-radius:14px !important;
    min-height:54px; justify-content:flex-start !important; text-align:left !important; transition: all .18s ease; }
[class*="st-key-sugg_"] button p { color: var(--green-900) !important; font-weight:700 !important; font-size:14px !important; text-align:left; }
[class*="st-key-sugg_"] button p, [class*="st-key-sugg_"] button [data-testid="stMarkdownContainer"] {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height:1.3; }
[class*="st-key-sugg_"] button > div, [class*="st-key-sugg_"] button span { justify-content:flex-start !important; text-align:left !important; width:100%; }
[class*="st-key-sugg_"] button:hover { background: var(--green-100) !important; border-style: solid !important; transform: translateY(-2px); }
.st-key-newchat button { background:#fff !important; border:1.5px solid #dfe8d8 !important; border-radius:14px !important; }
.st-key-newchat button p { color: var(--muted) !important; font-weight:800 !important; }

/* Chat input */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--cream) !important; }
.st-key-chatcard [data-testid="stChatInput"] { margin-top:12px; }
[data-testid="stChatInput"] { border:2px solid var(--green-600) !important; border-radius:18px !important; background:#fff !important;
    box-shadow: 0 10px 24px -16px rgba(23,61,36,.5); }
[data-testid="stChatInput"] textarea { font-size:16px !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--green-700) !important; border-radius:12px !important; }
[data-testid="stChatInputSubmitButton"] svg { fill:#fff !important; color:#fff !important; }

/* Sources panel */
.sources-panel { background:#fff; border:1px solid #e3ecd9; border-radius:26px; padding:18px; min-height:240px;
                 box-shadow: 0 12px 30px -20px rgba(23,61,36,.4); }
.src-head { font-weight:900; color:var(--green-900); font-size:16px; margin-bottom:12px; }
.src-card { border:1px solid #e6eedf; border-radius:16px; padding:12px 14px; margin-bottom:10px; background:#fcfdf9; animation: rise .4s ease-out both; }
.src-card:nth-child(3) { animation-delay:.08s } .src-card:nth-child(4) { animation-delay:.16s } .src-card:nth-child(5) { animation-delay:.24s }
.src-top { display:flex; gap:6px; align-items:center; margin-bottom:6px; flex-wrap:wrap; }
.src-num { background: var(--green-700); color:#fff; font-weight:900; width:24px; height:24px; border-radius:8px;
           display:inline-flex; align-items:center; justify-content:center; font-size:13px; }
.badge { font-size:11.5px; font-weight:800; padding:3px 9px; border-radius:999px; }
.badge.crop { background:#fff4d6; color:#7a5a12; } .badge.rw { background:#e3f0ff; color:#1d4f91; }
.src-title { font-weight:800; color:var(--ink); font-size:14px; line-height:1.35; }
.src-meta { font-size:12.5px; color:var(--muted); margin-top:4px; display:flex; gap:8px; flex-wrap:wrap; }
.src-link { color: var(--green-700) !important; font-weight:800; text-decoration:none; }
.src-card summary { cursor:pointer; font-size:12.5px; font-weight:800; color:var(--soil); margin-top:8px; }
.src-card details p { font-size:12.5px; color:#4a564e; line-height:1.5; background:#fff; border-radius:10px; padding:10px;
                      border:1px solid #eef3e8; max-height:200px; overflow:auto; }
.src-empty { text-align:center; color:var(--muted); padding: 22px 10px; } .src-empty b { color: var(--green-900); font-size:16px; }
.src-empty-icon { font-size:38px; margin-bottom:6px; } .spin { display:inline-block; animation: grow 1.2s ease-in-out infinite; }
@keyframes grow { 0%,100% { transform: scale(.85) } 50% { transform: scale(1.15) } }
.tips { background: linear-gradient(135deg,#fff7e0,#fffdf5); border:1px solid #f1e3b3; border-radius:22px; padding:16px 18px; margin-top:14px; }
.tips-head { font-weight:900; color:#6b4f12; margin-bottom:6px; } .tips ul { margin:0; padding-left:18px; color:#5b4a22; font-size:14px; line-height:1.6; }
.trusted { margin-top:10px; font-size:12.5px; color:#7b6a3e; }
.foot { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:12.5px;
        border-top:1px solid #e3ecd9; margin-top:22px; padding-top:14px; }
.credits { margin-top:8px; font-size:12px; color:var(--muted); } .credits summary { cursor:pointer; font-weight:800; }
.credits a { color: var(--green-700); }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
@media (max-width: 820px) {
  .hero { min-height: 0; border-radius:24px; }
  .hero-bg { background-position: 65% 75%; }
  .hero-shade { background: linear-gradient(180deg, rgba(12,36,20,.10) 0%, rgba(12,36,20,.55) 30%, rgba(12,36,20,.90) 60%); }
  .hero-content { padding: 150px 20px 22px; } .hero h1 { font-size:29px !important; }
  .photo-tag { top:14px; bottom:auto; } .status { display:none; }
  .stats div { min-width:0; flex:1; padding:8px 10px; } .stats b { font-size:18px; }
  .st-key-croprow [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; flex-direction: row !important; gap: 10px !important; }
  .st-key-croprow [data-testid="stColumn"] { width: calc(33.3% - 7px) !important; flex: 0 0 calc(33.3% - 7px) !important; min-width: calc(33.3% - 7px) !important; }
  [class*="st-key-crop_"] button { height:84px; padding:0 9px 8px !important; } [class*="st-key-crop_"] button p { font-size:13.5px !important; }
}
</style>""".replace("TILES", "\n".join(tiles)).replace("st-key-crop_SELECTED", f"st-key-crop_{selected}")


def header_html():
    return f"""
<header class="topbar"><div class="brand">{LOGO_SVG}<div><div class="brand-name">{APP_NAME}</div>
<div class="brand-sub">Smart Farming Assistant</div></div></div>
<div class="status"><span class="dot"></span> Online · {N_DOCS} trusted guides</div></header>
<section class="hero">
  <div class="hero-bg" style="background-image:url('{photo("hero")}')"></div><div class="hero-shade"></div>
  <div class="hero-content">
    <span class="pill">{RW_FLAG} Made for farmers in Rwanda</span>
    <h1>Good advice for every <span class="hl">season</span>, right when you need it.</h1>
    <p>Ask about pests, diseases, planting, fertilizer or storage. Every answer comes from agricultural
    guides you can trust, with the page it came from.</p>
    <div class="stats"><div><b>{N_DOCS}</b><span>trusted guides</span></div><div><b>5</b><span>main crops</span></div>
    <div><b>100%</b><span>answers with sources</span></div></div>
  </div>
  <div class="photo-tag">📍 Terraced hills, Rwanda</div>
</section>"""


TIPS = """<div class="tips"><div class="tips-head">💡 Tips for better answers</div><ul>
<li>Pick your crop above to search only its guides.</li><li>Describe what you see: colour, holes, spots, wilting.</li>
<li>Ask one question at a time.</li></ul>
<div class="trusted">Knowledge from <b>RAB</b> · <b>FAO</b> · <b>CIP</b> · <b>IITA</b> · <b>CIAT</b> · <b>CABI Plantwise</b></div></div>"""

WELCOME = f"""<div class="welcome"><div class="welcome-emoji">👋🏾</div><h3>Muraho! How can I help your farm today?</h3>
<p>Ask in simple words, for example <i>"my maize leaves have holes"</i>.<br>I answer only from {N_DOCS} trusted guides
from RAB, FAO, CIP and other experts, and I show you the source.</p></div>"""


def footer_html():
    items = "".join(f'<li>{html.escape(c["what"].title())}: <a href="{c["page"]}" target="_blank" rel="noopener">'
                    f'{html.escape(c["title"])}</a> by {html.escape(c["artist"])}, '
                    f'<a href="{c["license_url"]}" target="_blank" rel="noopener">{html.escape(c["license"])}</a></li>'
                    for c in CREDITS)
    return f"""<div class="foot"><span>🌾 Umurima helps you learn. For pesticides and doses, always confirm with your
local extension officer or RAB.</span><span>Built with Retrieval-Augmented Generation · NLP course project</span></div>
<details class="credits"><summary>📷 Photo credits (Wikimedia Commons)</summary><ul>{items}</ul></details>"""


# ----------------------------------------------------------------- state
st.session_state.setdefault("messages", [])     # [{"role", "content", "sources"}]
st.session_state.setdefault("crop", None)
st.session_state.setdefault("pending", None)

st.markdown(css(), unsafe_allow_html=True)
st.markdown(header_html(), unsafe_allow_html=True)

if not get_api_key() and not os.environ.get("UMURIMA_FAKE_LLM"):
    st.warning("The app is missing its Groq API key. Add **GROQ_API_KEY** in the app's Settings → Secrets.", icon="🔑")

# ----------------------------------------------------------------- crop tiles
st.markdown('<div class="section-title">Which crop is your question about?</div>', unsafe_allow_html=True)
with st.container(key="croprow"):
    for col, (label, crop) in zip(st.columns(6), CROPS):
        with col:
            if st.button(label, key=f"crop_{crop or 'all'}", use_container_width=True):
                st.session_state.crop = crop
                st.rerun()

# ----------------------------------------------------------------- main area
left, right = st.columns([3, 2], gap="large")
with right:
    last = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    sources_box = st.empty()
    sources_box.markdown(f'<div class="sources-panel">{last["sources"] if last else SOURCES_IDLE}</div>',
                         unsafe_allow_html=True)
    st.markdown(TIPS, unsafe_allow_html=True)

with left:
    with st.container(key="chatcard"):
        chatbox = st.container(height=520, key="chatbox", border=False)
        typed = st.chat_input("Type your farming question…")   # inside the card, so it scrolls with the page
        with chatbox:
            if not st.session_state.messages:
                st.markdown(WELCOME, unsafe_allow_html=True)
            for i, m in enumerate(st.session_state.messages):
                with st.chat_message(m["role"], avatar="🧑🏾‍🌾" if m["role"] == "user" else "🌱"):
                    marker = '<span class="u-msg"></span>' if m["role"] == "user" else '<span class="a-msg"></span>'
                    st.markdown(marker + m["content"], unsafe_allow_html=True)
                    if m["role"] == "assistant":
                        st.feedback("thumbs", key=f"fb_{i}")
        st.markdown('<div class="suggest-title">Try asking:</div>', unsafe_allow_html=True)
        qs = SUGGESTIONS[st.session_state.crop]
        for row in (qs[:2], qs[2:]):
            for col, q in zip(st.columns(2), row):
                with col:
                    if st.button(q, key=f"sugg_{qs.index(q)}", use_container_width=True):
                        st.session_state.pending = q
        if st.session_state.messages:
            if st.button("🔄 New chat", key="newchat"):
                st.session_state.messages = []
                st.rerun()

st.markdown(footer_html(), unsafe_allow_html=True)

question = (typed or st.session_state.pending or "").strip()
st.session_state.pending = None

# ----------------------------------------------------------------- answer a question
if question:
    st.session_state.messages.append({"role": "user", "content": html.escape(question)})
    with chatbox:
        with st.chat_message("user", avatar="🧑🏾‍🌾"):
            st.markdown('<span class="u-msg"></span>' + html.escape(question), unsafe_allow_html=True)
        sources_box.markdown(f'<div class="sources-panel">{SOURCES_SEARCHING}</div>', unsafe_allow_html=True)
        results = retrieve(question, st.session_state.crop)
        with st.chat_message("assistant", avatar="🌱"):
            try:
                answer = st.write_stream(stream_answer(build_messages(question, results)))
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if "rate" in msg and "limit" in msg:
                    answer = "⏳ Too many questions right now. Please wait a minute and try again."
                elif "api key" in msg or "api_key" in msg or "401" in msg or "auth" in msg:
                    answer = "🔑 The app's Groq API key is missing or wrong. Please tell the app owner."
                else:
                    answer = f"⚠️ Sorry, something went wrong while writing the answer ({type(e).__name__}). Please try again."
                st.markdown(answer)
                results = []
    answer = clean_answer(answer if isinstance(answer, str) else "".join(answer))
    if results and re.search(r"pesticide|fungicide|insecticide|herbicide|spray|chemical", answer, re.I) \
            and REFUSAL not in answer.lower():
        answer += ("\n\n<sub>⚠️ Before using any chemical, confirm the product and dose with your local "
                   "extension officer or RAB.</sub>")
    st.session_state.messages.append({"role": "assistant", "content": answer,
                                      "sources": sources_html(answer, results) if results else SOURCES_IDLE})
    st.rerun()