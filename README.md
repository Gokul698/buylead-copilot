# 🤝 BuyLead Copilot — AI matching for B2B buyer requirements

On a B2B marketplace, buyers post requirements like:

> *"bhai 300 jute bags chahiye with printing kolkata"*
> *"Need 2000 nos 3 ply box 12x10x8, Chennai delivery, max 15 rs urgent"*

These are short, messy, often Hinglish, and missing structure. Suppliers waste time on leads they can't serve, and buyers wait for replies.

**BuyLead Copilot** turns a messy requirement into a structured lead, finds the best-fit suppliers from a catalog using vector search, and drafts a quote reply. It then **checks the draft for hallucinated numbers** before it can be sent.

## How it works

```
Buyer text ─► 1. Extract ─► 2. Retrieve ─► 3. Rank ─► 4. Draft ─► 5. Verify
              (LLM JSON +    (TF-IDF or     (relevance,  (LLM or     (every ₹ / qty / day
               rule check)    embeddings)    city, MOQ,   template)   must exist in catalog)
                                             budget)
```

| Step | What it does | AI concept |
|---|---|---|
| Extract | LLM converts text to JSON (product, qty, city, budget, specs, urgency). A system prompt + a few-shot example guide it. Numbers the LLM returns are cross-checked against the original text, so it cannot invent a quantity or budget. | Structured prompting, few-shot, output validation |
| Retrieve | Supplier catalog is indexed as vectors. The lead is matched with cosine similarity. TF-IDF by default (no download); toggle to `all-MiniLM-L6-v2` sentence embeddings. | Embeddings, vector search (RAG retrieval) |
| Rank | Score = 60% relevance + 15% same city + 10% MOQ fits + 10% within budget + 5% rating. Irrelevant hits are dropped, not forced. | Business-rule re-ranking |
| Draft | LLM writes a reply using **only** the retrieved supplier row (grounded generation). | RAG generation |
| Verify | Every number in the draft must trace back to the catalog row or a value derived from it. Unverified numbers block the draft. | Hallucination guard |

**Works without an API key.** If `GROQ_API_KEY` is not set, extraction and drafting fall back to rule-based code. This keeps the demo free and fast (low token cost and latency).

## Results (eval/test_leads.csv, 15 labelled leads, rule-based mode)

| Metric | Score |
|---|---|
| Hit@1 | 100% |
| Hit@3 | 100% |
| MRR | 1.00 |
| Hallucination-guard pass | 100% |

The test set is small and hand-made, so these numbers show the pipeline works end to end. They are not a production benchmark. The guard was also tested by changing a price in a draft (₹14 → ₹11); it correctly flagged `11` as unverified.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py          # web UI
python -m eval.evaluate       # metrics
```

Optional LLM mode (free Groq key from console.groq.com):

```bash
export GROQ_API_KEY=your_key          # Windows: set GROQ_API_KEY=your_key
```

## Project structure

```
app.py               Streamlit UI
src/pipeline.py      extract → retrieve → rank → draft → verify
src/llm.py           Groq wrapper with graceful fallback
data/suppliers.csv   30 sample suppliers across 12 categories (synthetic)
eval/                labelled test leads + evaluation script
```

## Product thinking

- **Problem:** buyer leads are unstructured, so matching is slow and noisy.
- **Who benefits:** buyers get faster, relevant replies; suppliers only see leads they can fulfil (MOQ, city, budget).
- **Metric to track in production:** reply rate within 1 hour, and lead → order conversion.
- **Deliberately not built:** auto-sending replies. A human confirms, because a wrong price sent to a buyer costs trust.
- **Next steps:** Hindi/Tamil language support, a vector DB (FAISS/pgvector) for millions of listings, and feedback from accepted quotes to tune the ranking weights.

*Catalog data is synthetic, created for demonstration.*
