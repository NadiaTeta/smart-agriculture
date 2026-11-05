# Umurima: Smart Farming Assistant

A Retrieval-Augmented Generation (RAG) chatbot for smallholder farmers in Rwanda.
It answers questions about maize, beans, potatoes, cassava and bananas using only
23 trusted farming guides (RAB, FAO, CIP, IITA, CIAT, CABI Plantwise), and shows the
guide and page each answer comes from.

**How it works**
1. The question is turned into a vector with `BAAI/bge-small-en-v1.5`.
2. The 4 most similar chunks (out of 1,324) are found by cosine similarity.
3. The chunks and the question go to an LLM on Groq (`openai/gpt-oss-120b`), which
   answers only from them and cites the passages.

## What's in this folder

| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python libraries the app needs |
| `data/chunks.json` | The 1,324 text chunks from the 23 guides |
| `data/embeddings.npy` | The vector for each chunk (built in the Colab notebook) |
| `data/documents.json` | The list of guides with their links |
| `data/system_prompt.txt` | The rules the model must follow |
| `assets/` | Photos (Wikimedia Commons) and their credits |
| `.streamlit/config.toml` | Colours and theme |

## Deploy it (about 15 minutes)

### 1. Get a free Groq API key
1. Go to **console.groq.com** and sign in (Google works).
2. Open **API Keys** → **Create API Key**, name it "umurima", and copy the key (it starts with `gsk_`).
3. Keep it private. Never put it in the code or on GitHub.

### 2. Put the folder on GitHub
1. Go to **github.com** → **New repository**. Name it `umurima`, keep it **Public**, and click **Create repository**.
2. On the new repository page, click **uploading an existing file**.
3. Drag **everything inside this folder** (app.py, requirements.txt, README.md, and the `data`, `assets` and `.streamlit` folders) into the page.
   - The `.streamlit` folder may be hidden on your computer. If it doesn't upload, that's fine: it only sets the colours.
4. Click **Commit changes**.

### 3. Deploy on Streamlit Community Cloud
1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Choose your `umurima` repository, branch `main`, main file `app.py`.
4. Open **Advanced settings**:
   - **Python version:** 3.12
   - **Secrets:** paste this, with your real key:
     ```
     GROQ_API_KEY = "gsk_your_key_here"
     ```
5. Choose an app address (e.g. `umurima`) and click **Deploy**.

The first start takes a few minutes while it installs the libraries. After that, share the link with your team.

## Good to know
- **Sleeping:** free apps go to sleep when nobody uses them for a while. The next visitor clicks "wake up" and waits a short time.
- **Limits:** Groq's free plan limits how many questions per minute and per day. If you see "Too many questions right now", wait a minute.
- **Changing the model:** add `GROQ_MODEL = "openai/gpt-oss-20b"` (or another Groq model) in Secrets. No code change needed.
- **Updating the guides:** rebuild the index in the Colab notebook and replace the files in `data/`.
- **Testing on your own computer:** `pip install -r requirements.txt`, put the key in `.streamlit/secrets.toml`, then `streamlit run app.py`.

## Credits
Photos from Wikimedia Commons under free licences; see "Photo credits" at the bottom of the app.
Farming guides belong to their publishers and are linked from every answer.
