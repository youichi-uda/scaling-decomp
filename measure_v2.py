#!/usr/bin/env python
"""Per-token-class loss measurement (v2) for one Pythia model size.

Adds Web Claude's make-or-break control: WORD-level frequency matching.
Each predicted token is tagged with the frequency of the whitespace-delimited WORD it
belongs to, so we can compare word-initial vs continuation loss WITHIN matched word
frequency (token-level matching does not equalize word predictability).

Outputs JSON aggregations:
  by_position, by_depth, by_freqdecile_position (token freq),
  by_wordfreqdecile_position (WORD freq, the control), by_seqband_position.

Usage: python measure_v2.py --model EleutherAI/pythia-410m --docs 200 --seqlen 1024 --out v2_410m.json
"""
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def build_vocab_flags(tok):
    vocab_size = len(tok)
    toks = tok.convert_ids_to_tokens(list(range(vocab_size)))
    starts_space = np.zeros(vocab_size, dtype=bool)
    body_alpha = np.zeros(vocab_size, dtype=bool)
    bodies = [""] * vocab_size
    for i, t in enumerate(toks):
        if t is None:
            continue
        ss = t.startswith("Ġ") or t.startswith("Ċ")
        body = t[1:] if ss else t
        starts_space[i] = ss
        body_alpha[i] = len(body) > 0 and body.isalpha()
        bodies[i] = body
    return starts_space, body_alpha, bodies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="NeelNanda/pile-10k")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--revision", default=None, help="HF revision (e.g. step10000 for Pythia checkpoints)")
    args = ap.parse_args()
    dev = "cuda"

    print(f"[{args.model}] tokenizer/dataset (rev={args.revision})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision) if args.revision else AutoTokenizer.from_pretrained(args.model)
    starts_space, body_alpha, bodies = build_vocab_flags(tok)

    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    else:
        ds = load_dataset(args.dataset, split=args.dataset_split)
    texts = ds[args.text_field][: args.docs]
    all_ids = []
    for t in texts:
        all_ids.extend(tok(t, return_attention_mask=False)["input_ids"])
    n_chunks = len(all_ids) // args.seqlen
    flat = np.array(all_ids[: n_chunks * args.seqlen], dtype=np.int64)  # [T]
    T = flat.shape[0]
    print(f"[{args.model}] {n_chunks} chunks x {args.seqlen} = {T} tokens", flush=True)

    # ---- token-level metadata over the flat stream ----
    tok_ss = starts_space[flat]          # starts with space
    tok_alpha = body_alpha[flat]
    # word id: new word whenever a token starts with space (or first token)
    new_word = tok_ss.copy(); new_word[0] = True
    word_id = np.cumsum(new_word) - 1     # [T]
    n_words = int(word_id[-1]) + 1

    # within-word depth
    depth = np.zeros(T, dtype=np.int32)
    d = 0
    for i in range(T):
        d = 0 if new_word[i] else d + 1
        depth[i] = d

    # reconstruct word strings -> word frequency
    word_strs = [None] * n_words
    cur = []
    cur_wid = 0
    for i in range(T):
        if new_word[i] and i > 0:
            word_strs[cur_wid] = "".join(cur)
            cur = []
            cur_wid = word_id[i]
        cur.append(bodies[flat[i]])
    word_strs[cur_wid] = "".join(cur)
    from collections import Counter
    wc = Counter(word_strs)
    word_freq_by_wid = np.array([wc[word_strs[w]] for w in range(n_words)], dtype=np.int64)
    tok_word_freq = word_freq_by_wid[word_id]   # [T] freq of the word each token belongs to

    # token-level frequency (per occurrence)
    uniq, counts = np.unique(flat, return_counts=True)
    id2c = np.zeros(starts_space.shape[0], dtype=np.int64); id2c[uniq] = counts
    tok_freq = id2c[flat]

    # ---- run model, per-token loss ----
    print(f"[{args.model}] model", flush=True)
    mkw = {"revision": args.revision} if args.revision else {}
    # HF Hub bug: Pythia 2.8B/12B model.safetensors files are mis-registered at some
    # revisions (early steps point to main weights). Force pytorch_model.bin via
    # use_safetensors=False to ensure revision-correct weights.
    if args.revision and any(x in args.model for x in ("pythia-2.8b","pythia-12b")):
        mkw["use_safetensors"] = False
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **mkw).to(dev).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **mkw).to(dev).eval()
    chunks = flat.reshape(n_chunks, args.seqlen)
    losses = np.empty((n_chunks, args.seqlen - 1), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_chunks, args.bs):
            batch = torch.from_numpy(chunks[i : i + args.bs]).to(dev)
            logits = model(batch).logits[:, :-1, :].float()
            labels = batch[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
            ).reshape(batch.size(0), -1)
            losses[i : i + batch.size(0)] = ce.cpu().numpy()
            if (i // args.bs) % 10 == 0:
                print(f"  {i}/{n_chunks}", flush=True)
    losses = losses.reshape(-1)  # [n_chunks*(seqlen-1)]

    # predicted-token global indices: chunk r, col c (0..seqlen-2) -> global r*seqlen + c+1
    rows = np.repeat(np.arange(n_chunks), args.seqlen - 1)
    cols = np.tile(np.arange(args.seqlen - 1), n_chunks)
    gidx = rows * args.seqlen + cols + 1   # index into flat metadata for the PREDICTED token

    p_ss = tok_ss[gidx]; p_alpha = tok_alpha[gidx]
    p_depth = depth[gidx]; p_wfreq = tok_word_freq[gidx]; p_tfreq = tok_freq[gidx]
    p_col = cols

    # word-length (number of subwords in each word) — for the multi-token-only control
    word_lengths = np.bincount(word_id, minlength=n_words)
    p_wordlen = word_lengths[word_id[gidx]]
    word_initial = p_ss & p_alpha
    word_cont = (~p_ss) & p_alpha
    other = ~p_alpha

    def agg(mask):
        m = mask & np.isfinite(losses)
        return {"n": int(m.sum()), "mean_loss": float(losses[m].mean()) if m.any() else None}

    # decile bin helper on a value array restricted to a base mask
    def decile_bins(values, base_mask):
        v = values[base_mask]
        qs = np.quantile(v, np.linspace(0, 1, 11))
        return qs

    res = {
        "model": args.model, "n_pred_tokens": int(losses.shape[0]),
        "overall_mean_loss": float(np.nanmean(losses)),
        "by_position": {"word_initial": agg(word_initial), "word_continuation": agg(word_cont), "other": agg(other)},
        "by_depth": {"0_initial": agg(word_initial),
                     "1": agg(word_cont & (p_depth == 1)), "2": agg(word_cont & (p_depth == 2)),
                     "3plus": agg(word_cont & (p_depth >= 3))},
        "by_seqband_position": {
            "early_lt128": {"word_initial": agg(word_initial & (p_col < 128)), "word_continuation": agg(word_cont & (p_col < 128))},
            "mid": {"word_initial": agg(word_initial & (p_col >= 128) & (p_col < args.seqlen - 256)),
                    "word_continuation": agg(word_cont & (p_col >= 128) & (p_col < args.seqlen - 256))},
            "late": {"word_initial": agg(word_initial & (p_col >= args.seqlen - 256)),
                     "word_continuation": agg(word_cont & (p_col >= args.seqlen - 256))},
        },
        "by_freqdecile_position": {}, "by_wordfreqdecile_position": {},
        # WEB CLAUDE OBJECTION 2 CONTROL: restrict to multi-token words only (word_len >= 2),
        # so cont and init come from the SAME word set.
        "by_position_multitokenonly": {
            "word_initial": agg(word_initial & (p_wordlen >= 2)),
            "word_continuation": agg(word_cont & (p_wordlen >= 2)),
        },
        "by_wordlen_position": {
            "len2_init": agg(word_initial & (p_wordlen == 2)),
            "len2_cont": agg(word_cont & (p_wordlen == 2)),
            "len3_init": agg(word_initial & (p_wordlen == 3)),
            "len3_cont": agg(word_cont & (p_wordlen == 3)),
            "len4plus_init": agg(word_initial & (p_wordlen >= 4)),
            "len4plus_cont": agg(word_cont & (p_wordlen >= 4)),
        },
        # multi-token-only × word-freq decile (the strict control)
        "by_wordfreqdecile_position_multitokenonly": {},
    }
    # token-freq deciles over alpha tokens
    tq = decile_bins(p_tfreq, word_initial | word_cont)
    tdec = np.clip(np.digitize(p_tfreq, tq[1:-1]), 0, 9)
    for b in range(10):
        dm = tdec == b
        res["by_freqdecile_position"][str(b)] = {"freq_range": [float(tq[b]), float(tq[b + 1])],
            "word_initial": agg(word_initial & dm), "word_continuation": agg(word_cont & dm)}
    # WORD-freq deciles over alpha tokens (the control)
    wq = decile_bins(p_wfreq, word_initial | word_cont)
    wdec = np.clip(np.digitize(p_wfreq, wq[1:-1]), 0, 9)
    for b in range(10):
        dm = wdec == b
        res["by_wordfreqdecile_position"][str(b)] = {"wordfreq_range": [float(wq[b]), float(wq[b + 1])],
            "word_initial": agg(word_initial & dm), "word_continuation": agg(word_cont & dm)}

    # Multi-token-only word-freq deciles (deciles computed within the multi-token-word subset only,
    # so the word set is held fixed in both classes within each decile)
    mt_mask = (word_initial | word_cont) & (p_wordlen >= 2)
    if mt_mask.any():
        mwq = decile_bins(p_wfreq, mt_mask)
        mwdec = np.clip(np.digitize(p_wfreq, mwq[1:-1]), 0, 9)
        for b in range(10):
            dm = (mwdec == b) & (p_wordlen >= 2)
            res["by_wordfreqdecile_position_multitokenonly"][str(b)] = {
                "wordfreq_range": [float(mwq[b]), float(mwq[b + 1])],
                "word_initial": agg(word_initial & dm), "word_continuation": agg(word_cont & dm)}

    # ---- EXACT WORD-OCCURRENCE MATCHED AGGREGATION (Codex objection 2) ----
    # For each multi-token word occurrence: one (init_loss, mean_cont_loss) pair.
    # Each word contributes equally regardless of how many cont tokens it has.
    # This is the "same-word-set held fixed" estimand the paper claims.
    pred_wid = word_id[gidx]   # word id of each predicted token
    # find multi-token words (length >= 2 subwords)
    multitoken_word_mask_per_word = word_lengths >= 2
    # for each predicted token, is it init of a multi-token word, or cont of a multi-token word?
    is_mt_init = word_initial & (p_wordlen >= 2)
    is_mt_cont = word_cont & (p_wordlen >= 2)
    # Per-word init loss: each multi-token word has at most one init position
    # (because we use word_id, and predicted positions are unique)
    per_word_init_loss = {}
    per_word_cont_losses = {}
    per_word_freq = {}
    for idx in np.flatnonzero(is_mt_init | is_mt_cont):
        w = int(pred_wid[idx])
        if idx in np.flatnonzero(is_mt_init): pass  # alternative selector kept simple below
    # vectorized: build per-word lists efficiently via groupby on word id
    mt_idx = np.flatnonzero(is_mt_init | is_mt_cont)
    mt_w   = pred_wid[mt_idx]
    mt_loss = losses[mt_idx]
    mt_init_flag = is_mt_init[mt_idx]
    mt_wfreq = p_wfreq[mt_idx]
    # group: per word id, list of (loss, is_init, wfreq)
    order = np.argsort(mt_w, kind="stable")
    mt_w_s = mt_w[order]; mt_loss_s = mt_loss[order]
    mt_init_flag_s = mt_init_flag[order]; mt_wfreq_s = mt_wfreq[order]
    unique_w, group_starts = np.unique(mt_w_s, return_index=True)
    group_ends = np.r_[group_starts[1:], len(mt_w_s)]
    per_word_records = []
    for gs, ge in zip(group_starts, group_ends):
        init_mask = mt_init_flag_s[gs:ge]
        if init_mask.any():
            init_loss_w = float(mt_loss_s[gs:ge][init_mask].mean())
        else:
            init_loss_w = None
        cont_mask = ~init_mask
        if cont_mask.any():
            cont_mean_w = float(mt_loss_s[gs:ge][cont_mask].mean())
        else:
            cont_mean_w = None
        wfreq_w = float(mt_wfreq_s[gs])  # all same word, same wfreq
        per_word_records.append((init_loss_w, cont_mean_w, wfreq_w))
    # Aggregate per-word records by word-freq decile (deciles computed over per-word wfreq)
    wfreqs = np.array([r[2] for r in per_word_records])
    if len(wfreqs):
        per_word_dec_q = np.quantile(wfreqs, np.linspace(0, 1, 11))
        per_word_dec = np.clip(np.digitize(wfreqs, per_word_dec_q[1:-1]), 0, 9)
    else:
        per_word_dec = np.array([])
    res["by_wordfreqdecile_position_per_word"] = {}
    for b in range(10):
        bin_words = [per_word_records[i] for i in range(len(per_word_records)) if per_word_dec[i] == b]
        init_losses = [r[0] for r in bin_words if r[0] is not None]
        cont_losses = [r[1] for r in bin_words if r[1] is not None]
        res["by_wordfreqdecile_position_per_word"][str(b)] = {
            "wordfreq_range": [float(per_word_dec_q[b]), float(per_word_dec_q[b + 1])] if len(wfreqs) else [None, None],
            "n_words": int(len(bin_words)),
            "word_initial": {
                "n_words": int(len(init_losses)),
                "mean_loss": float(np.mean(init_losses)) if init_losses else None,
            },
            "word_continuation": {
                "n_words": int(len(cont_losses)),
                "mean_loss": float(np.mean(cont_losses)) if cont_losses else None,
            },
        }
    # Also store the per-word array (compact) so we can re-aggregate without re-running
    init_arr = np.array([r[0] if r[0] is not None else np.nan for r in per_word_records], dtype=np.float32)
    cont_arr = np.array([r[1] if r[1] is not None else np.nan for r in per_word_records], dtype=np.float32)
    res["per_word_summary"] = {
        "n_multitoken_words": int(len(per_word_records)),
        "init_loss_mean": float(np.nanmean(init_arr)) if len(init_arr) else None,
        "cont_loss_mean": float(np.nanmean(cont_arr)) if len(cont_arr) else None,
        "init_loss_std": float(np.nanstd(init_arr)) if len(init_arr) else None,
        "cont_loss_std": float(np.nanstd(cont_arr)) if len(cont_arr) else None,
        "diff_mean": float(np.nanmean(init_arr - cont_arr)) if len(init_arr) else None,
        "diff_std": float(np.nanstd(init_arr - cont_arr)) if len(init_arr) else None,
    }

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[{args.model}] wrote {args.out}  overall={res['overall_mean_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
