# Architecture Justification: Encoder-only Transformer for Extractive QA

## Chosen model

**deepset/roberta-base-squad2** — a RoBERTa (Robustly Optimised BERT Pretraining Approach)
encoder fine-tuned on SQuAD 2.0.

---

## Why an encoder-only model (BERT/RoBERTa) instead of a decoder-only model (GPT)?

### The task demands span extraction, not generation

Extractive QA is a *span selection* task: given a context paragraph and a question,
the model must identify the exact start and end token positions of the answer inside
the context. The output is a substring of the input — it is never invented.

Encoder-only models are built for exactly this:

| Property | Encoder-only (BERT/RoBERTa) | Decoder-only (GPT) |
|---|---|---|
| Attention direction | **Bidirectional** — every token attends to every other token | **Causal / left-to-right** — each token only sees past tokens |
| Pre-training objective | Masked Language Modelling (MLM) | Next-token prediction (CLM) |
| Output head for QA | Two linear layers predicting start/end span | Generate tokens autoregressively |
| Hallucination risk | None — answer must come from context | Can generate text not in the context |
| Inference speed | Single forward pass | Sequential token generation |

For a support-ticket system, producing an answer that is **guaranteed to come from the
ticket itself** is a hard requirement. A decoder-only model can and does fabricate
plausible-sounding component names that are not present in the ticket.

---

## How self-attention enables contextual understanding

Self-attention allows each token in the sequence to compute a weighted sum of
representations of **all other tokens** simultaneously:

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V
```

For the query *"What is the primary technical component mentioned?"* and a context like
*"The user reported that the Windows 10 driver for the NVIDIA RTX 3070 causes a BSOD..."*,
self-attention lets the token `NVIDIA` directly attend to `driver`, `Windows`, and `BSOD`
in a single layer — capturing the dependency that `NVIDIA RTX 3070` is the component
responsible for the described failure. An LSTM would have to pass this signal through
many sequential hidden states, losing it over distance.

Multi-head attention runs `h` independent attention functions in parallel, each learning
a different relationship (syntactic, semantic, coreference). The representations are
concatenated and projected, giving the model a rich, multi-faceted view of each token.

---

## How positional encoding preserves word order

Self-attention is inherently **order-agnostic**: the attention formula treats the input
as a set, not a sequence. Without positional information, "the driver crashed Windows"
and "Windows crashed the driver" would produce identical representations.

BERT-family models inject a learned **positional embedding** added to each token embedding
before the first attention layer:

```
input_i = token_embedding_i + positional_embedding_i
```

This allows the model to distinguish "NVIDIA driver" (component → attribute) from
"driver NVIDIA" (inverted) and to understand that an answer near position 400 of a
500-token ticket is structurally different from one at position 10.

---

## Why BERT/RoBERTa outperforms LSTMs on this task

| Capability | LSTM | Transformer (encoder) |
|---|---|---|
| Long-range dependencies | Degrades with sequence length (vanishing gradient path) | Constant-distance attention between any two tokens |
| Parallelism during training | Sequential — O(n) operations | Fully parallel — O(1) sequential operations |
| Contextual representations | Hidden state carries compressed history | Every token sees the full context at every layer |
| Transfer learning | Rarely pretrained at scale | Pretrained on billions of tokens; fine-tuned cheaply |

For support tickets that can be several paragraphs long, a component name near the
end of the ticket may be the answer to a question whose keywords appear only at the
start. Self-attention handles this trivially; an LSTM would require the signal to
survive many time steps through a bottleneck hidden state.

---

## Why SQuAD 2.0 fine-tuning matters

SQuAD 2.0 includes questions that are **unanswerable** given the context. Fine-tuning on
this dataset teaches the model to output a low-confidence score and an empty span rather
than always forcing an answer. In a real support-ticket system, some tickets may not
mention a specific component, so this "abstain when unsure" behaviour directly reduces
false positives.

---

## Summary

RoBERTa fine-tuned on SQuAD 2.0 is chosen because:

1. Bidirectional self-attention captures long-range dependencies that LSTMs miss.
2. Positional encoding preserves word order without sequential computation.
3. Span-extraction output is grounded in the source text, eliminating hallucination.
4. SQuAD 2.0 fine-tuning handles unanswerable questions gracefully.
5. A single forward pass is orders of magnitude faster than autoregressive generation.
