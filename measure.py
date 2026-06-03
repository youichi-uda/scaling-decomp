#!/usr/bin/env python
"""Per-token-class loss measurement for one Pythia model size.

Outputs a JSON with mean CE loss aggregated by:
  - position class (word-initial / word-continuation / other)
  - within-word depth (0,1,2,3+)
  - frequency decile x position class  (the frequency-matched control)

Frequency deciles are derived from the eval corpus itself with a FIXED tokenizer,
so they are identical across model sizes (the tokenizer is shared in the Pythia suite).

Usage: python measure.py --model EleutherAI/pythia-70m --docs 200 --seqlen 1024 --out out_70m.json
"""
import argparse, json, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def build_vocab_flags(tok):
    """For each token id: (starts_space, body_alpha)."""
    vocab_size = len(tok)
    ids = list(range(vocab_size))
    toks = tok.convert_ids_to_tokens(ids)
    starts_space = np.zeros(vocab_size, dtype=bool)
    body_alpha = np.zeros(vocab_size, dtype=bool)
    for i, t in enumerate(toks):
        if t is None:
            continue
        ss = t.startswith("Ġ") or t.startswith("Ċ")
        body = t[1:] if ss else t
        starts_space[i] = ss
        body_alpha[i] = len(body) > 0 and body.isalpha()
    return starts_space, body_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="NeelNanda/pile-10k")
    args = ap.parse_args()

    dev = "cuda"
    print(f"[{args.model}] loading tokenizer/dataset", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    starts_space, body_alpha = build_vocab_flags(tok)

    ds = load_dataset(args.dataset, split="train")
    texts = ds["text"][: args.docs]

    # Tokenize into fixed-length chunks.
    all_ids = []
    for t in texts:
        enc = tok(t, return_attention_mask=False)["input_ids"]
        all_ids.extend(enc)
    n_chunks = len(all_ids) // args.seqlen
    all_ids = all_ids[: n_chunks * args.seqlen]
    chunks = np.array(all_ids, dtype=np.int64).reshape(n_chunks, args.seqlen)
    print(f"[{args.model}] {n_chunks} chunks x {args.seqlen} = {chunks.size} tokens", flush=True)

    print(f"[{args.model}] loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    losses = []   # per predicted token
    tgt_ids = []
    bs = 8
    with torch.no_grad():
        for i in range(0, n_chunks, bs):
            batch = torch.from_numpy(chunks[i : i + bs]).to(dev)
            logits = model(batch).logits  # [b, L, V]
            # predict token t+1 from <=t  => targets are batch[:,1:]
            shift_logits = logits[:, :-1, :].float()
            shift_labels = batch[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            )
            losses.append(ce.detach().to("cpu").numpy())
            tgt_ids.append(shift_labels.detach().to("cpu").numpy().reshape(-1))
            if (i // bs) % 5 == 0:
                print(f"  {i}/{n_chunks}", flush=True)

    losses = np.concatenate(losses)        # [P]
    tgt_ids = np.concatenate(tgt_ids)      # [P]
    P = losses.shape[0]
    print(f"[{args.model}] {P} predicted tokens, mean loss {losses.mean():.4f}", flush=True)

    # Position class of each PREDICTED (target) token.
    pred_ss = starts_space[tgt_ids]
    pred_alpha = body_alpha[tgt_ids]
    word_initial = pred_ss & pred_alpha
    word_cont = (~pred_ss) & pred_alpha
    other = ~pred_alpha

    # Within-word depth: tokens since last starts_space (per chunk).
    depth = np.zeros(P, dtype=np.int32)
    # reconstruct per-chunk sequence of target starts_space
    ss_seq = pred_ss.reshape(n_chunks, args.seqlen - 1)
    depth_seq = np.zeros_like(ss_seq, dtype=np.int32)
    for r in range(n_chunks):
        d = 0
        for c in range(args.seqlen - 1):
            if ss_seq[r, c]:
                d = 0
            else:
                d += 1
            depth_seq[r, c] = d
    depth = depth_seq.reshape(-1)

    # Frequency deciles from this corpus (per-occurrence frequency of target id).
    uniq, counts = np.unique(tgt_ids, return_counts=True)
    id2count = np.zeros(starts_space.shape[0], dtype=np.int64)
    id2count[uniq] = counts
    occ_freq = id2count[tgt_ids].astype(np.float64)  # frequency of each occurrence's token
    # decile boundaries on the per-occurrence frequency distribution
    qs = np.quantile(occ_freq, np.linspace(0, 1, 11))
    freq_dec = np.clip(np.digitize(occ_freq, qs[1:-1]), 0, 9)

    # Sequence-position band (control for early-sequence high loss).
    col_idx = np.tile(np.arange(args.seqlen - 1), n_chunks)
    seq_early = col_idx < 128
    seq_mid = (col_idx >= 128) & (col_idx < args.seqlen - 256)
    seq_late = col_idx >= args.seqlen - 256

    def agg(mask):
        m = mask & np.isfinite(losses)
        return {"n": int(m.sum()), "mean_loss": float(losses[m].mean()) if m.any() else None}

    result = {
        "model": args.model,
        "n_pred_tokens": int(P),
        "overall_mean_loss": float(losses.mean()),
        "by_position": {
            "word_initial": agg(word_initial),
            "word_continuation": agg(word_cont),
            "other": agg(other),
        },
        "by_depth": {
            "0_initial": agg(word_initial),
            "1": agg(word_cont & (depth == 1)),
            "2": agg(word_cont & (depth == 2)),
            "3plus": agg(word_cont & (depth >= 3)),
        },
        "by_freqdecile_position": {},
        "by_seqband_position": {
            "early_lt128": {"word_initial": agg(word_initial & seq_early), "word_continuation": agg(word_cont & seq_early)},
            "mid": {"word_initial": agg(word_initial & seq_mid), "word_continuation": agg(word_cont & seq_mid)},
            "late": {"word_initial": agg(word_initial & seq_late), "word_continuation": agg(word_cont & seq_late)},
        },
    }
    for dbin in range(10):
        dm = freq_dec == dbin
        result["by_freqdecile_position"][str(dbin)] = {
            "freq_range": [float(qs[dbin]), float(qs[dbin + 1])],
            "word_initial": agg(word_initial & dm),
            "word_continuation": agg(word_cont & dm),
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.model}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
