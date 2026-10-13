"""Streamlit UI.  Run:  streamlit run app.py"""
import streamlit as st

from src.llm import llm_available
from src.pipeline import SupplierIndex, run

st.set_page_config(page_title="BuyLead Copilot", page_icon="🤝", layout="wide")
st.title("🤝 BuyLead Copilot")
st.caption("Messy B2B buyer requirement → structured lead → best-fit suppliers → verified quote draft")


@st.cache_resource
def get_index(use_emb):
    return SupplierIndex("data/suppliers.csv", use_embeddings=use_emb)


with st.sidebar:
    st.subheader("Settings")
    use_emb = st.toggle("Semantic embeddings (sentence-transformers)", value=False)
    k = st.slider("Candidates to retrieve", 3, 10, 5)
    st.markdown(f"**LLM:** {'Groq ✅' if llm_available() else 'Off – rule-based fallback'}")
    st.caption("Set GROQ_API_KEY in your environment to enable LLM extraction and drafting.")

examples = ["Need 2000 nos 3 ply box 12x10x8 for ecommerce, Chennai delivery, max 15 rs urgent",
            "bhai 300 jute bags chahiye with printing kolkata",
            "quote for 100 ISI safety helmets for construction site thane"]
text = st.text_area("Buyer requirement", examples[0], height=90)
st.caption("Try: " + " | ".join(examples[1:]))

if st.button("Find suppliers", type="primary") and text.strip():
    index = get_index(use_emb)
    r = run(text, index, k=k)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("1 · Structured lead")
        st.json(r["lead"])
    with c2:
        st.subheader(f"2 · Ranked suppliers ({index.mode})")
        if r["matches"].empty:
            st.warning("No relevant supplier in catalog — flag lead for manual sourcing.")
        else:
            st.dataframe(r["matches"][["supplier_name", "city", "product", "price_inr", "unit",
                                       "min_order_qty", "same_city", "moq_ok", "in_budget",
                                       "similarity", "score"]], use_container_width=True)
    if r["draft"]:
        st.subheader("3 · Draft quote reply (top supplier)")
        st.write(r["draft"])
        if r["check"]["passed"]:
            st.success("Guard passed: every number in the draft traces back to the catalog.")
        else:
            st.error(f"Guard failed: unverified numbers {r['check']['unverified_numbers']} — do not send.")
