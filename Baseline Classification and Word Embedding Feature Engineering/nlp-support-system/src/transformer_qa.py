"""
transformer_qa.py
Phase 3 – Extractive Question Answering using a pre-trained Transformer model.

Model  : deepset/roberta-base-squad2  (encoder-only, fine-tuned on SQuAD 2.0)
Task   : Given a support-ticket as context, extract the primary technical
         component mentioned using the Hugging Face `pipeline` API.
Output : reports/qa_outputs.json
"""

import os
import json
import pandas as pd
from transformers import pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data",    "all_tickets_processed_improved_v3.csv")
OUTPUT_PATH  = os.path.join(BASE_DIR, "reports", "qa_outputs.json")

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

# ── Question asked of every ticket ────────────────────────────────────────────
QUESTION = "what is the department should this ticket be assigned to ?"

# ── 1. Load the dataset and sample a small test set ───────────────────────────
def load_test_set(file_path: str, n_samples: int = 10, random_state: int = 42) -> list[dict]:
    """
    Load the CSV and return n_samples tickets formatted as QA examples.
    We specifically pick at least 2 tickets where the answer is likely far from
    the beginning of the text (proxy: tickets longer than 300 characters).
    """
    df = pd.read_csv(file_path)

    # Keep only rows with non-null ticket text
    df = df.dropna(subset=["Document"]).reset_index(drop=True)

    # Separate long tickets (answer likely far from the start) and short ones
    long_tickets  = df[df["Document"].str.len() > 300].sample(
        min(3, len(df[df["Document"].str.len() > 300])),
        random_state=random_state
    )
    short_tickets = df[df["Document"].str.len() <= 300].sample(
        max(0, n_samples - len(long_tickets)),
        random_state=random_state
    )

    test_df = pd.concat([long_tickets, short_tickets]).reset_index(drop=True)

    test_set = []
    for _, row in test_df.iterrows():
        test_set.append({
            "id":       str(row.name),
            "topic":    str(row.get("Topic_group", "Unknown")),
            "context":  str(row["Document"]),
            "question": QUESTION,
        })
    return test_set


# ── 2. Load the pre-trained QA pipeline ───────────────────────────────────────
def load_qa_pipeline(model_name: str = "deepset/roberta-base-squad2"):
    """
    Load an extractive QA pipeline from Hugging Face.
    deepset/roberta-base-squad2 is chosen because:
      - Encoder-only (bidirectional) attention gives better span extraction than
        decoder-only models (see architecture_justification.md)
      - Fine-tuned on SQuAD 2.0, which includes "unanswerable" questions —
        important for tickets that may not mention a specific component.
    """
    print(f"Loading model: {model_name}")
    qa = pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
        # Handle long contexts by striding the context window
        max_seq_len=512,
        doc_stride=128,
    )
    print("Model loaded.\n")
    return qa


# ── 3. Run inference ──────────────────────────────────────────────────────────
def run_qa(qa_pipeline, test_set: list[dict]) -> list[dict]:
    """
    Run the QA pipeline on every example and collect results.
    """
    results = []
    for i, example in enumerate(test_set):
        print(f"[{i+1}/{len(test_set)}] Ticket ID: {example['id']}  |  Topic: {example['topic']}")
        print(f"  Context (first 120 chars): {example['context'][:120]}...")

        prediction = qa_pipeline(
            question=example["question"],
            context=example["context"],
        )

        result = {
            "ticket_id":      example["id"],
            "topic_group":    example["topic"],
            "question":       example["question"],
            "context_length": len(example["context"]),
            "answer":         prediction["answer"],
            "score":          round(prediction["score"], 4),
            # Character position of the answer inside the context
            "answer_start":   prediction["start"],
            "answer_end":     prediction["end"],
            # Flag tickets where the answer is far from the start (long-range dependency demo)
            "long_range":     prediction["start"] > 150,
            "context_preview": example["context"][:300],
        }

        print(f"  Answer: '{result['answer']}'  (score={result['score']}, start_char={result['answer_start']})")
        if result["long_range"]:
            print(f"  ** Long-range dependency: answer begins at char {result['answer_start']} **")
        print()
        results.append(result)

    return results


# ── 4. Save results ───────────────────────────────────────────────────────────
def save_results(results: list[dict], output_path: str):
    summary = {
        "model":         "deepset/roberta-base-squad2",
        "question":      QUESTION,
        "total_tickets": len(results),
        "long_range_examples": sum(1 for r in results if r["long_range"]),
        "avg_score":     round(sum(r["score"] for r in results) / len(results), 4),
        "results":       results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1 – build test set
    test_set = load_test_set(DATA_PATH, n_samples=10)
    print(f"Test set: {len(test_set)} tickets ({sum(1 for t in test_set if len(t['context']) > 300)} long-context)\n")

    # Step 2 – load model
    qa = load_qa_pipeline()

    # Step 3 – inference
    results = run_qa(qa, test_set)

    # Step 4 – save
    save_results(results, OUTPUT_PATH)

    # Step 5 – highlight the long-range examples explicitly
    long_range = [r for r in results if r["long_range"]]
    if len(long_range) >= 2:
        print("\n=== Long-range dependency demonstrations ===")
        for r in long_range[:2]:
            print(f"\nTicket {r['ticket_id']} | Topic: {r['topic_group']}")
            print(f"  Context length  : {r['context_length']} chars")
            print(f"  Answer          : '{r['answer']}'")
            print(f"  Answer position : chars {r['answer_start']}–{r['answer_end']}")
            print(f"  Confidence      : {r['score']}")
    else:
        print("\nNote: fewer than 2 long-range examples found — increase n_samples or lower the threshold.")
