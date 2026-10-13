"""BuyLead Copilot pipeline:
messy buyer requirement -> structured lead -> retrieve suppliers (vector search)
-> rank -> draft quote reply -> verify draft against catalog (hallucination guard)."""
import re
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import chat

CITIES = ["ahmedabad", "chennai", "delhi", "coimbatore", "rajkot", "pune", "mumbai", "tiruppur",
          "ludhiana", "surat", "bengaluru", "bangalore", "noida", "haridwar", "jaipur", "hyderabad",
          "kolkata", "indore", "mysuru", "udupi", "kochi", "faridabad", "thane", "gurugram",
          "moradabad", "salem", "vapi", "manipal", "mangaluru", "madurai", "lucknow"]
UNITS = r"(pieces|piece|pcs|nos|units|kg|kgs|litres|litre|liters|ltr|bags|bag|meters|meter|mtr|sq ft|coils|coil|sets|set|pairs|pair|tons|ton)"


@dataclass
class Lead:
    product: str = ""
    quantity: float | None = None
    unit: str = ""
    city: str = ""
    budget_inr: float | None = None
    specs: list = field(default_factory=list)
    urgency: str = "normal"
    source: str = "rules"


# ---------- 1. Extraction ----------
EXTRACT_SYS = """You convert messy B2B buyer requirements (English/Hinglish) into JSON.
Return ONLY JSON with keys: product (short noun phrase), quantity (number or null),
unit (string), city (string or ""), budget_inr (number per unit or null),
specs (list of short strings), urgency ("urgent" or "normal").
Never invent values that are not in the text; use null/"" instead.

Example input: "need 2000 nos 3ply box 12x10x8 chennai delivery, max 15 rs, urgent"
Example output: {"product":"3 ply corrugated box","quantity":2000,"unit":"piece","city":"chennai",
"budget_inr":15,"specs":["12x10x8 inch"],"urgency":"urgent"}"""


def extract_rules(text: str) -> Lead:
    t = text.lower()
    lead = Lead()
    m = re.search(r"(\d[\d,]*\.?\d*)\s*" + UNITS, t)
    if m:
        lead.quantity = float(m.group(1).replace(",", ""))
        lead.unit = m.group(2)
    lead.city = next((c for c in CITIES if re.search(rf"\b{c}\b", t)), "")
    b = re.search(r"(?:budget|max|under|upto|up to|@|rs\.?|₹|inr)\s*(?:rs\.?|₹|inr)?\s*(\d[\d,]*)", t)
    if b:
        lead.budget_inr = float(b.group(1).replace(",", ""))
    lead.urgency = "urgent" if re.search(r"urgent|asap|immediate|jaldi|today|tomorrow", t) else "normal"
    lead.specs = re.findall(r"\d+\s*(?:x\s*\d+\s*)+(?:inch|mm|cm)?|\d+\s*(?:hp|w|watt|gsm|ply|micron|mm|sqmm|inch)\b", t)
    # product = text with numbers / noise words stripped
    noise = r"\b(need|required|require|want|looking for|urgent|asap|please|pls|send|quote|quotation|price|delivery|deliver|to|in|at|for|with|max|budget|rs|inr|per|piece|nos|kg|ki|ka|chahiye|hai|bhai|sir)\b"
    prod = re.sub(r"\d[\d,.]*", " ", t)
    prod = re.sub(noise, " ", prod)
    prod = re.sub("|".join(lead.city and [lead.city] or ["$^"]), " ", prod)
    lead.product = " ".join(w for w in re.sub(r"[^a-z ]", " ", prod).split() if len(w) > 1)[:80]
    return lead


def extract(text: str) -> Lead:
    data = chat(EXTRACT_SYS, text, json_mode=True)
    if not data:
        return extract_rules(text)
    rules = extract_rules(text)  # cross-check: LLM must not invent numbers absent from text
    lead = Lead(**{k: data.get(k) for k in asdict(Lead()) if k in data and k != "source"}, source="llm")
    nums = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)}
    if lead.quantity is not None and float(lead.quantity) not in nums:
        lead.quantity = rules.quantity
    if lead.budget_inr is not None and float(lead.budget_inr) not in nums:
        lead.budget_inr = rules.budget_inr
    lead.city = (lead.city or "").lower()
    return lead


# ---------- 2. Retrieval (vector search) ----------
class SupplierIndex:
    """TF-IDF vectors by default (no download, runs anywhere).
    Set use_embeddings=True to use sentence-transformers semantic embeddings."""

    def __init__(self, csv_path: str, use_embeddings: bool = False):
        self.df = pd.read_csv(csv_path)
        self.docs = (self.df["product"] + ". " + self.df["category"] + ". " + self.df["description"]).tolist()
        self.mode = "tfidf"
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.vecs = self.model.encode(self.docs, normalize_embeddings=True)
                self.mode = "embeddings"
                return
            except Exception as e:
                print(f"[index] embeddings unavailable, using TF-IDF: {e}")
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.vecs = self.vec.fit_transform(self.docs)

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        if self.mode == "embeddings":
            q = self.model.encode([query], normalize_embeddings=True)
            sims = (self.vecs @ q.T).ravel()
        else:
            sims = (self.vecs @ self.vec.transform([query]).T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        out = self.df.iloc[idx].copy()
        out["similarity"] = sims[idx].round(3)
        return out


# ---------- 3. Ranking ----------
def rank(lead: Lead, cands: pd.DataFrame) -> pd.DataFrame:
    c = cands.copy()
    c["same_city"] = c["city"].str.lower() == (lead.city or "").lower()
    c["moq_ok"] = True if lead.quantity is None else c["min_order_qty"] <= lead.quantity
    c["in_budget"] = True if lead.budget_inr is None else c["price_inr"] <= lead.budget_inr
    c["score"] = (0.60 * c["similarity"] / max(c["similarity"].max(), 1e-9)
                  + 0.15 * c["same_city"] + 0.10 * c["moq_ok"] + 0.10 * c["in_budget"]
                  + 0.05 * (c["rating"] / 5)).round(3)
    c = c[c["similarity"] > 0.05]  # drop irrelevant hits instead of forcing a match
    return c.sort_values("score", ascending=False).reset_index(drop=True)


# ---------- 4. Draft reply ----------
DRAFT_SYS = """You write a short, polite quotation reply from a supplier to a buyer on a B2B marketplace.
Use ONLY facts from the SUPPLIER ROW (price, unit, MOQ, lead time, city). Do not invent discounts,
certifications or prices. Max 90 words. Plain text."""


def draft_reply(lead: Lead, row: pd.Series, buyer_text: str) -> str:
    facts = row[["supplier_name", "city", "product", "description", "price_inr", "unit",
                 "min_order_qty", "lead_time_days"]].to_dict()
    out = chat(DRAFT_SYS, f"BUYER: {buyer_text}\nSUPPLIER ROW: {facts}", temperature=0.3)
    if out:
        return out.strip()
    qty = f"{int(lead.quantity)} {lead.unit}" if lead.quantity else "your required quantity"
    total = f" (approx. ₹{int(row.price_inr * lead.quantity):,} for {qty})" if lead.quantity else ""
    return (f"Hello, thank you for your enquiry. {row.supplier_name}, {row.city} can supply "
            f"{row['product']} at ₹{row.price_inr:g} per {row.unit}{total}. "
            f"Minimum order: {row.min_order_qty} {row.unit}. Dispatch in {row.lead_time_days} days. "
            f"Please confirm and share delivery details.")


# ---------- 5. Verification (hallucination guard) ----------
def verify(draft: str, row: pd.Series, lead: Lead) -> dict:
    """Every rupee figure / day count in the draft must be traceable to the catalog row or
    a value derived from it. Flags confident-but-wrong outputs before they reach a buyer."""
    allowed = {float(row.price_inr), float(row.min_order_qty), float(row.lead_time_days)}
    if lead.quantity:
        allowed |= {float(lead.quantity), float(row.price_inr * lead.quantity)}
    nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", draft)]
    spec_nums = {float(n) for n in re.findall(r"\d+\.?\d*", str(row.description) + " " + str(row["product"]))}
    unknown = [n for n in nums if n not in allowed and n not in spec_nums]
    return {"passed": not unknown, "unverified_numbers": unknown}


def run(buyer_text: str, index: SupplierIndex, k: int = 5) -> dict:
    lead = extract(buyer_text)
    query = " ".join([lead.product] + list(lead.specs)) or buyer_text
    ranked = rank(lead, index.search(query, k=k))
    result = {"lead": asdict(lead), "matches": ranked, "draft": None, "check": None}
    if len(ranked):
        top = ranked.iloc[0]
        result["draft"] = draft_reply(lead, top, buyer_text)
        result["check"] = verify(result["draft"], top, lead)
    return result
