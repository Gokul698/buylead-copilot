"""Measures retrieval quality (Hit@1, Hit@3, MRR) and the hallucination-guard pass rate.
Run:  python -m eval.evaluate"""
import pandas as pd

from src.pipeline import SupplierIndex, run

index = SupplierIndex("data/suppliers.csv")
tests = pd.read_csv("eval/test_leads.csv")
rows = []
for _, t in tests.iterrows():
    r = run(t.buyer_text, index)
    ids = r["matches"]["supplier_id"].tolist()
    rank = ids.index(t.expected_supplier) + 1 if t.expected_supplier in ids else None
    rows.append({"lead": t.buyer_text[:45], "expected": t.expected_supplier,
                 "top1": ids[0] if ids else "-", "rank": rank,
                 "guard_pass": r["check"]["passed"] if r["check"] else None})

res = pd.DataFrame(rows)
print(res.to_string(index=False))
print(f"\nHit@1: {(res['rank'] == 1).mean():.0%}   Hit@3: {(res['rank'] <= 3).mean():.0%}   "
      f"MRR: {res['rank'].apply(lambda x: 1 / x if x else 0).mean():.2f}   "
      f"Guard pass: {res['guard_pass'].mean():.0%}")
