"""tools/expert_partition_study.py - the model, not the engine: can the routed experts be split over two cards so
that few tokens need the other card, and what does that cost in bytes?

Everything here is arithmetic over routing traces: no engine, no GPU, no download. A trace comes from
`--dump-routing` (record = int32 layer, int32 k, k int32 ids, k float32 weights): one record per position per layer.

THE MODEL
  * Every (layer, expert) pair gets one label: card 0, card 1, or the CPU. A card's labels must fit its capacity in
    bytes; the CPU has no limit (it is the RAM/disk tier the engine already has).
  * One event = one token at one layer, with the k experts the router picked (k = 10 here). The event is computed on
    the card that holds most of those k; every expert of the event that sits on the OTHER card is a jump.
  * A jump costs 2 x n_embd x dtype bytes: one activation row out and one result row back. That row is the same for
    every remote expert of the layer, so the cost is per event, not per expert - which is what the peer tier does
    (include/strata/core/peer_experts.hpp: n_tok rows of n_embd floats). `--jump-per-expert` prices the pessimistic
    variant where every remote expert pays its own row.
  * An expert on the CPU costs its blob from RAM plus the activation rows, once per lookup.

THE TWO QUESTIONS, SEPARATELY
  A. With the same experts on the cards, can a smarter split cut the jumps? `rank` splits each layer down its
     ranked list, alternating the cards (the shape of a stripe); `co-act` splits the same set by co-activation
     (greedy + Kernighan-Lin, with a knapsack over layers for how much of each layer sits on card 0).
  B. Which experts should be on the cards at all? `--on-card quota` keeps the same share of every layer (the
     engine's per-layer quota, slots / n_layers); `--on-card global` takes the hottest pairs overall, however they
     fall across layers. Both hold the same total bytes.
  `layer` puts whole layers on a card - every expert of a layer, as opposed to a quota of it - and is shown for
  scale: it is what a layer split would hold if a card had room for whole layers.

Use:
  python tools/expert_partition_study.py TRACEDIR [--pack DIR] [--card-gib 18] [--out partitions.npz]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_LAYER, N_EXPERT = 48, 512
N_EMBD = 2560          # Qwen3.8-Flash-Next, from the pack index: token_embd.weight is 2560 x 248320


def load_traces(tracedir):
    files = sorted(Path(tracedir).glob("c*.trace"))
    if not files:
        raise SystemExit(f"{tracedir}: no c*.trace")
    per_file = []
    for f in files:
        raw = np.fromfile(f, dtype=np.int32)
        k = int(raw[1])
        rec = 2 + 2 * k
        n = raw.size // rec
        r = raw[: n * rec].reshape(-1, rec)
        layer, ids = r[:, 0].astype(np.int32), r[:, 2:2 + k].astype(np.int16)
        per_file.append((f.stem, {int(l): ids[layer == l] for l in np.unique(layer)}))
    return per_file


def load_blobs(packdir):
    path = Path(packdir) / "native_experts.txt"
    blobs = np.zeros(N_LAYER, dtype=np.int64)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            blobs[int(p[0])] = int(p[4])
    return blobs


def events_of(per_file):
    """per-layer event matrix over every trace, and the trace index of every event."""
    events, which = {}, {}
    for t, (_, layers) in enumerate(per_file):
        for l, ids in layers.items():
            events.setdefault(l, []).append(ids)
            which.setdefault(l, []).append(np.full(len(ids), t, dtype=np.int32))
    return ({l: np.concatenate(v) for l, v in events.items()},
            {l: np.concatenate(v) for l, v in which.items()})


def layer_cost(ids, lab, blob, k, jump_unit, cpu_unit, per_expert=False, home=None):
    L = lab[ids]
    c0 = (L == 0).sum(1)
    c1 = (L == 1).sum(1)
    cu = (L == 2).sum(1)
    if home is not None:
        # one home card for the residual: the token never changes card between layers, so every event that needs an
        # expert of the OTHER card pays one round trip (the activation row out and the result back)
        remote = c1 if home == 0 else c0
    else:
        remote = (k - cu) - np.maximum(c0, c1)
    jumps = remote > 0
    jump = int(jumps.sum()) * jump_unit * (int(remote[jumps].sum()) if per_expert else 1)
    cpu = int((cu * (blob + cpu_unit)).sum())
    return jump, cpu, int(jumps.sum()), int(cu.sum())


def totals(labels, events, blobs, k, jump_unit, cpu_unit, per_expert=False, domains=None, home=None):
    out = {"jump": 0, "cpu": 0, "jumps": 0, "cpu_lookups": 0, "events": 0}
    per_dom = {}
    for l, ids in events.items():
        j, c, nj, nc = layer_cost(ids, labels[l], int(blobs[l]), k, jump_unit, cpu_unit, per_expert, home)
        out["jump"] += j
        out["cpu"] += c
        out["jumps"] += nj
        out["cpu_lookups"] += nc
        out["events"] += len(ids)
        if domains is not None:
            d = domains[l]
            for name in np.unique(d):
                m = d == name
                jj, cc, njj, _ = layer_cost(ids[m], labels[l], int(blobs[l]), k, jump_unit, cpu_unit, per_expert, home)
                e = per_dom.setdefault(int(name), {"jump": 0, "cpu": 0, "jumps": 0, "events": 0})
                e["jump"] += jj
                e["cpu"] += cc
                e["jumps"] += njj
                e["events"] += int(m.sum())
    return out, per_dom


def used_bytes(labels, blobs):
    return [sum(int((labels[l] == c).sum()) * int(blobs[l]) for l in range(N_LAYER)) for c in (0, 1)]


def freq_rank(events):
    return {l: np.bincount(ids.ravel(), minlength=N_EXPERT) for l, ids in events.items()}


def cooc(ids):
    """Co-activation counts: C[i, j] = how many events routed both i and j. 45 pairs an event, 10 experts."""
    C = np.zeros((N_EXPERT, N_EXPERT), dtype=np.int32)
    for row in ids:
        u = np.unique(row)
        if len(u) < 2:
            continue
        a, b = np.triu_indices(len(u), 1)
        np.add.at(C, (u[a], u[b]), 1)
    C += C.T
    return C


def score_split(lab, ids, k, jump_unit):
    """Events of one layer that need both cards (the true objective, no proxy)."""
    L = lab[ids]
    return int(((L == 0).any(1) & (L == 1).any(1)).sum()) * jump_unit


def greedy_split(Csub, order, n0):
    """Assign the experts (compact indices, hottest first) to card 0 (n0 of them) and card 1, each to the side it is
    more strongly co-activated with; a full side takes no more."""
    n = Csub.shape[0]
    lab = np.ones(n, dtype=np.int8)
    in0 = np.zeros(n, dtype=bool)
    left0, left1 = n0, n - n0
    for i in order:
        if left0 == 0:
            break
        if left1 == 0:
            lab[i], in0[i] = 0, True
            left0 -= 1
            continue
        w0 = Csub[i][in0].sum()
        w1 = Csub[i][~in0].sum()
        if w0 > w1 or (w0 == w1 and left0 >= left1):
            lab[i], in0[i] = 0, True
            left0 -= 1
        else:
            left1 -= 1
    return lab


def kl_refine(lab, Csub, rounds=40):
    """Kernighan-Lin: swap a pair between the cards while the cut falls. The counts stay the same, so the cards'
    bytes do not change."""
    for _ in range(rounds):
        A = np.flatnonzero(lab == 0)
        B = np.flatnonzero(lab == 1)
        if len(A) == 0 or len(B) == 0:
            break
        dA = Csub[np.ix_(A, A)].sum(1) - Csub[np.ix_(A, B)].sum(1)
        dB = Csub[np.ix_(B, B)].sum(1) - Csub[np.ix_(B, A)].sum(1)
        M = dA[:, None] + dB[None, :] - 2 * Csub[np.ix_(A, B)]
        i, j = np.unravel_index(int(np.argmax(M)), M.shape)
        if M[i, j] <= 0:
            break
        lab[A[i]], lab[B[j]] = 1, 0
    return lab


def layer_split_curve(ids, freq, n_on, k, jump_unit, grid=9):
    """For one layer: J(n) = the fewest jump bytes with n of its on-card experts on card 0. The on-card set is the
    layer's `n_on` hottest pairs, as today's fill keeps them. Returns (ns, Js, on, Csub, rel)."""
    C = cooc(ids)
    order = np.argsort(-freq, kind="stable")
    on = order[:n_on]
    Csub = C[np.ix_(on, on)]
    rel = np.argsort(-freq[on], kind="stable")        # compact order, hottest first
    ns = sorted({int(round(x)) for x in np.linspace(0, n_on, grid)} | {0, n_on})
    J = []
    for n0 in ns:
        sub = kl_refine(greedy_split(Csub, rel, n0), Csub)
        lab = np.full(N_EXPERT, 2, dtype=np.int8)
        lab[on] = sub
        J.append(score_split(lab, ids, k, jump_unit))
    return ns, J, on, Csub, rel


def greedy_fill(events, blobs, caps, home):
    """The hottest pairs first: the home card until it is full, then the other card, the CPU for the rest."""
    freq = freq_rank(events)
    labels = np.full((N_LAYER, N_EXPERT), 2, dtype=np.int8)
    pairs = sorted(((int(freq[l][e]), l, e) for l in range(N_LAYER) for e in range(N_EXPERT)), key=lambda x: -x[0])
    free = list(caps)
    for _, l, e in pairs:
        for card in (home, 1 - home):
            if free[card] >= blobs[l]:
                labels[l][e] = card
                free[card] -= int(blobs[l])
                break
    return labels


def local_search(labels, events, blobs, caps, k, jump_unit, cpu_unit, per_expert, home, sweeps=4):
    """One (layer, expert) at a time, hottest first, while the whole-trajectory cost falls and the cards fit."""
    labels = labels.copy()
    used = used_bytes(labels, blobs)
    free = [caps[0] - used[0], caps[1] - used[1]]
    cur = {l: layer_cost(events[l], labels[l], int(blobs[l]), k, jump_unit, cpu_unit, per_expert, home)
           for l in range(N_LAYER)}
    for _ in range(sweeps):
        moved = 0
        for l in range(N_LAYER):
            ids, blob = events[l], int(blobs[l])
            for e in np.argsort(-np.bincount(ids.ravel(), minlength=N_EXPERT), kind="stable"):
                src, base = int(labels[l][e]), cur[l]
                for dst in (0, 1, 2):
                    if dst == src or (dst != 2 and free[dst] < blob):
                        continue
                    labels[l][e] = dst
                    cand = layer_cost(ids, labels[l], blob, k, jump_unit, cpu_unit, per_expert, home)
                    if cand[0] + cand[1] < base[0] + base[1]:
                        if src != 2:
                            free[src] += blob
                        if dst != 2:
                            free[dst] -= blob
                        cur[l] = base = cand
                        src = dst
                        moved += 1
                labels[l][e] = src
        if moved == 0:
            break
    return labels


def token_moves(labels, per_file):
    """The token follows its work: at each layer it computes on the card holding most of its experts, and a jump is a
    change of that card. Returns (moves, tokens) - the residual rows one card has to send to the other."""
    total_moves, total_tokens = 0, 0
    for _, layers in per_file:
        n = min(len(v) for v in layers.values())
        cur = np.full(n, -1)
        for l in range(N_LAYER):
            ids = layers.get(l)
            if ids is None:
                continue
            lab = labels[l][ids[:n]]
            c0 = (lab == 0).sum(1)
            c1 = (lab == 1).sum(1)
            m = np.where(c0 > c1, 0, np.where(c1 > c0, 1, -1))
            valid = m >= 0
            nxt = np.where(valid, m, cur)
            total_moves += int((valid & (cur >= 0) & (nxt != cur)).sum())
            cur = nxt
        total_tokens += n
    return total_moves, total_tokens


def fill_with_dupes(events, blobs, caps, ndup):
    """The `ndup` hottest pairs go on BOTH cards (label 3), then the next hottest fill card 0, then card 1, the CPU
    takes the rest. Duplication spends a card's room on purpose: a hot expert on both cards is one no token ever
    fetches or moves for."""
    freq = freq_rank(events)
    pairs = sorted(((int(freq[l][e]), l, e) for l in range(N_LAYER) for e in range(N_EXPERT)), key=lambda x: -x[0])
    labels = np.full((N_LAYER, N_EXPERT), 2, dtype=np.int8)
    free = list(caps)
    placed = 0
    for _, l, e in pairs:
        if placed >= ndup:
            break
        if free[0] >= blobs[l] and free[1] >= blobs[l]:
            labels[l][e] = 3
            free[0] -= int(blobs[l])
            free[1] -= int(blobs[l])
            placed += 1
    for card in (0, 1):
        for _, l, e in pairs:
            if labels[l][e] != 2:
                continue
            if free[card] >= blobs[l]:
                labels[l][e] = card
                free[card] -= int(blobs[l])
    return labels


def path_cost(labels, tok, blobs, row):
    """The token chooses, layer by layer: stay and fetch the experts only the other card has (one row each), or jump
    (one row) and continue there. A dynamic program over the two cards, per token. An expert on BOTH cards is never
    fetched - that is what duplication buys. Returns (bytes, jumps, cpu_bytes)."""
    T = tok.shape[0]
    dp0 = np.zeros(T, dtype=np.float64)
    dp1 = np.zeros(T, dtype=np.float64)
    mv0 = np.zeros((N_LAYER, T), dtype=bool)
    mv1 = np.zeros((N_LAYER, T), dtype=bool)
    cpu = 0
    for l in range(N_LAYER):
        lab = labels[l][tok[:, l, :]]
        r0 = (lab == 1).sum(1)          # experts only on card 1: fetched if the token stays on card 0
        r1 = (lab == 0).sum(1)
        cu = (lab == 2).sum(1)
        cpu += int((cu * int(blobs[l])).sum())
        stay0, stay1 = dp0 + r0 * row, dp1 + r1 * row
        move0, move1 = dp1 + row + r0 * row, dp0 + row + r1 * row
        mv0[l] = move0 < stay0
        mv1[l] = move1 < stay1
        dp0, dp1 = np.minimum(stay0, move0), np.minimum(stay1, move1)
    state = (dp0 > dp1).astype(np.int8)
    jumps = 0
    for l in range(N_LAYER - 1, -1, -1):
        jumped = np.where(state == 0, mv0[l], mv1[l])
        jumps += int(jumped.sum())
        state = np.where(jumped, 1 - state, state)
    return float(np.minimum(dp0, dp1).sum()), jumps, cpu


def token_matrix(per_file):
    """(T, 48, k) expert ids: one row a token, its 48 layers. Every trace writes one record per layer per position,
    in the same order, so position p of layer L is token p."""
    rows = []
    for _, layers in per_file:
        n = min(len(v) for v in layers.values())
        m = np.zeros((n, N_LAYER, next(iter(layers.values())).shape[1]), dtype=np.int16)
        for l, ids in layers.items():
            m[:, l, :] = ids[:n]
        rows.append(m)
    return np.concatenate(rows)


def jump_cost(labels, tok, blobs, row_bytes):
    """Jumps when the token continues where its work is: the card holding most of its experts, per layer, and a jump
    each time that card changes. Plus the CPU lookups. Returns (jump_bytes, cpu_bytes, jumps)."""
    T = tok.shape[0]
    cur = np.full(T, -1, dtype=np.int8)
    jumps, cpu = 0, 0
    for l in range(N_LAYER):
        lab = labels[l][tok[:, l, :]]
        c0 = (lab == 0).sum(1)
        c1 = (lab == 1).sum(1)
        cu = (lab == 2).sum(1)
        cpu += int((cu * int(blobs[l])).sum())
        m = np.where(c0 > c1, 0, np.where(c1 > c0, 1, -1)).astype(np.int8)
        valid = m >= 0
        nxt = np.where(valid, m, cur)
        jumps += int((valid & (cur >= 0) & (nxt != cur)).sum())
        cur = nxt
    return jumps * row_bytes, cpu, jumps


def optimize_jumps(labels, tok, blobs, caps, row_bytes, top=1500, sweeps=2):
    """Move one expert at a time, hottest first, while jumps + CPU traffic fall and the cards fit."""
    labels = labels.copy()
    used = used_bytes(labels, blobs)
    free = [caps[0] - used[0], caps[1] - used[1]]
    freq_all = np.zeros((N_LAYER, N_EXPERT), dtype=np.int64)
    for l in range(N_LAYER):
        flat = tok[:, l, :].ravel()
        freq_all[l] = np.bincount(flat, minlength=N_EXPERT)
    order = sorted(((int(freq_all[l][e]), l, e) for l in range(N_LAYER) for e in range(N_EXPERT)),
                   key=lambda x: -x[0])[:top]
    best = jump_cost(labels, tok, blobs, row_bytes)
    for sweep in range(sweeps):
        moved = 0
        for _, l, e in order:
            src = int(labels[l][e])
            for dst in (0, 1, 2):
                if dst == src or (dst != 2 and free[dst] < blobs[l]):
                    continue
                labels[l][e] = dst
                cand = jump_cost(labels, tok, blobs, row_bytes)
                if cand[0] + cand[1] < best[0] + best[1] - 1:
                    if src != 2:
                        free[src] += int(blobs[l])
                    if dst != 2:
                        free[dst] -= int(blobs[l])
                    best = cand
                    src = dst
                    moved += 1
            labels[l][e] = src
        print(f"    sweep {sweep}: jumps {best[2]}, jump {best[0] / 1e6:.1f} MB, CPU {best[1] / 1e6:.1f} MB, "
              f"moved {moved}", file=sys.stderr)
        if moved == 0:
            break
    return labels, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracedir")
    ap.add_argument("--pack", default="/data/Strata-data/packs/iq3_s")
    ap.add_argument("--corpus", default=str(ROOT / "bench" / "statistic_aware_corpus.json"))
    ap.add_argument("--card-gib", type=float, default=18.0)
    ap.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    ap.add_argument("--jump-per-expert", action="store_true")
    ap.add_argument("--on-card", choices=("quota", "global"), default="quota",
                    help="quota: every layer keeps the same share of the cards (the engine's per-layer quota); "
                         "global: the hottest pairs overall, however they fall across layers")
    ap.add_argument("--contiguous", action="store_true",
                    help="only contiguous layer ranges per card (what a layer split is), one hand-off per window, "
                         "instead of a free assignment that may alternate the cards layer by layer")
    ap.add_argument("--home", choices=("roam", "0", "1"), default="roam",
                    help="roam: each layer is computed on the card that holds most of its experts (the residual may "
                         "change card between layers); 0/1: one home card, the other only computes its experts' rows")
    ap.add_argument("--optimize-jumps", action="store_true",
                    help="search a free expert distribution (both cards hold every layer) for the fewest jumps "
                         "a token makes, against the CPU traffic; starts from the frequency fill")
    ap.add_argument("--top", type=int, default=1500, help="experts per sweep the jump search tries")
    ap.add_argument("--path-cost", action="store_true",
                    help="price a whole token path: stay and fetch the experts the other card has, or jump and "
                         "continue there (a dynamic program over the two cards)")
    ap.add_argument("--duplicate-sweep", default="",
                    help="comma-separated counts of the hottest pairs to put on BOTH cards")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    per_file = load_traces(a.tracedir)
    events, which = events_of(per_file)
    blobs = load_blobs(a.pack)
    k = next(iter(events.values())).shape[1]
    dtype_bytes = 4 if a.dtype == "f32" else 2
    jump_unit = 2 * N_EMBD * dtype_bytes
    cpu_unit = jump_unit
    cap = int(a.card_gib * (1 << 30))
    caps = [cap, cap]

    corpus = json.load(open(a.corpus, encoding="utf-8"))
    dom_names = sorted({c["d"] for c in corpus})
    dom_of = {i: dom_names.index(c["d"]) for i, c in enumerate(corpus)}
    domains = {l: np.array([dom_of.get(int(t), -1) for t in which[l]]) for l in which}

    freq = freq_rank(events)
    total_events = sum(len(v) for v in events.values())
    print(f"# {len(per_file)} traces, {total_events} events, k={k}, n_embd={N_EMBD}, {a.dtype}: a jump is {jump_unit} B")
    print(f"# experts {int(blobs.sum()) * N_EXPERT / 2**30:.1f} GiB over {N_LAYER}x{N_EXPERT} pairs; "
          f"two cards hold {2 * a.card_gib:.1f} GiB of it")

    # ---- the on-card set: per layer, the hottest N_l pairs, N_l by the layer's blob size, so every layer gets
    # the same share of the two cards (this is the engine's per-layer quota: slots / n_layers)
    n_on = np.zeros(N_LAYER, dtype=np.int32)
    if a.on_card == "quota":
        for l in range(N_LAYER):
            n_on[l] = int(min(N_EXPERT, (2 * cap) // (int(blobs[l]) * N_LAYER)))
    else:
        flat = np.concatenate([freq[l] for l in range(N_LAYER)])
        order = np.argsort(-flat, kind="stable")
        used, taken = 0, 0
        for idx in order:
            l, e = int(idx // N_EXPERT), int(idx % N_EXPERT)
            if used + int(blobs[l]) > 2 * cap:
                break
            used += int(blobs[l])
            n_on[l] += 1
            taken += 1
    T = sum(int(n_on[l]) * int(blobs[l]) for l in range(N_LAYER))
    print(f"# policy {a.on_card}: on the cards {T / 2**30:.2f} GiB of {2 * cap / 2**30:.1f} GiB, "
          f"{int(n_on.sum())} pairs ({int(n_on.sum()) / (N_LAYER * N_EXPERT) * 100:.1f}%), "
          f"{n_on.min()} to {n_on.max()} a layer; CPU keeps the rest")

    labels_rank = np.full((N_LAYER, N_EXPERT), 2, dtype=np.int8)
    labels_opt = np.full((N_LAYER, N_EXPERT), 2, dtype=np.int8)
    curves = {}
    for l in range(N_LAYER):
        if n_on[l] == 0:
            curves[l] = None
            continue
        ns, Js, on, Csub, rel = layer_split_curve(events[l], freq[l], int(n_on[l]), k, jump_unit, grid=9)
        curves[l] = (ns, Js, on, Csub, rel)
        rank_lab = np.full(N_EXPERT, 2, dtype=np.int8)
        rank_lab[on] = np.arange(len(on)) % 2          # today's shape: alternate down the ranked list
        labels_rank[l] = rank_lab

    # knapsack: how many of each layer's on-card pairs sit on card 0. Both cards must fit, and they should be
    # used evenly; the objective is the summed curve (fewest jumps).
    step = 16 << 20
    W = cap // step
    dp = np.full(W + 1, np.inf)
    dp[0] = 0.0
    choice = [None] * N_LAYER
    for l in range(N_LAYER):
        c = curves[l]
        if c is None:
            continue
        ns, Js = c[0], c[1]
        cost_b = np.array([int(n) * int(blobs[l]) for n in ns], dtype=np.float64) / step
        nd = np.full(W + 1, np.inf)
        pick = np.zeros(W + 1, dtype=np.int16)
        for wi in range(W + 1):
            if dp[wi] == np.inf:
                continue
            for j in range(len(ns)):
                wj = wi + int(round(cost_b[j]))
                if wj <= W and dp[wi] + Js[j] < nd[wj]:
                    nd[wj] = dp[wi] + Js[j]
                    pick[wj] = j
        dp = nd
        choice[l] = (ns, Js, pick)
    lo = max(0, min(W, int(np.ceil((T - cap) / step))))
    best_w, best_v = min(lo, W), np.inf
    for w in range(lo, W + 1):
        if dp[w] == np.inf:
            continue
        v = dp[w] + 1e9 * abs(w * step - T / 2) / max(T, 1)
        if v < best_v:
            best_v, best_w = v, w
    picks = [0] * N_LAYER
    wi = best_w
    for l in reversed(range(N_LAYER)):
        c = choice[l]
        if c is None:
            continue
        ns, Js, pick = c
        j = int(pick[wi])
        picks[l] = j
        wi = max(0, wi - int(round(int(ns[j]) * int(blobs[l]) / step)))
    for l in range(N_LAYER):
        c = curves[l]
        if c is None:
            continue
        ns, Js, on, Csub, rel = c
        n0 = int(ns[picks[l]])
        lab = np.full(N_EXPERT, 2, dtype=np.int8)
        lab[on] = kl_refine(greedy_split(Csub, rel, n0), Csub)
        labels_opt[l] = lab

    # ---- whole layers on one card, the layer split: each card splits its room over the layers it holds
    labels_layer = np.full((N_LAYER, N_EXPERT), 2, dtype=np.int8)
    order_l = sorted(range(N_LAYER), key=lambda x: -int(freq[x].sum()) * int(blobs[x]))
    room = list(caps)
    owner = {}
    for l in order_l:
        card = 0 if room[0] >= room[1] else 1
        if room[card] < N_EXPERT * int(blobs[l]):
            continue                                   # not even the whole layer fits: the CPU takes all of it
        room[card] -= N_EXPERT * int(blobs[l])
        owner[l] = card
    for card in (0, 1):
        mine = [l for l in owner if owner[l] == card]
        if not mine:
            continue
        budget = int(caps[card])
        total_b = sum(int(blobs[l]) for l in mine)
        for l in mine:
            quota = int(budget * int(blobs[l]) / total_b) // int(blobs[l])
            quota = max(0, min(N_EXPERT, quota))
            top = np.argsort(-freq[l], kind="stable")[:quota]
            labels_layer[l][top] = card

    # ---- the same question under the real constraint: card 0 runs layers 0..K-1, card 1 runs K..47. Then a token
    # crosses cards once per pass, not once per layer - the hand-off the free assignment above charges nothing for.
    handoff_tokens = (5 * N_EMBD + 4) * dtype_bytes      # handoff_floats = hc*n_embd + n_embd + hc, hc = 4 -> 5 rows
    n_tokens = total_events // N_LAYER
    if a.contiguous:
        best = None
        for K in range(1, N_LAYER):
            labels = np.full((N_LAYER, N_EXPERT), 2, dtype=np.int8)
            ok = True
            for card, lo, hi in ((0, 0, K), (1, K, N_LAYER)):
                budget = cap
                pairs = sorted(((int(freq[l][e]), l, e) for l in range(lo, hi) for e in range(N_EXPERT)),
                               key=lambda x: -x[0])
                for _, l, e in pairs:
                    if budget < blobs[l]:
                        break
                    budget -= int(blobs[l])
                    labels[l][e] = card
            t, _ = totals(labels, events, blobs, k, jump_unit, cpu_unit, a.jump_per_expert)
            u = used_bytes(labels, blobs)
            if max(u) > cap:
                ok = False
            score = t["cpu"] + handoff_tokens * n_tokens          # one hand-off per window
            if ok and (best is None or score < best[0]):
                best = (score, K, t, u)
        score, K, t, u = best
        share = (t["events"] - t["jumps"]) / t["events"] * 100
        print("")
        print(f"contiguous (layers 0..{K - 1} on card 0, {K}..{N_LAYER - 1} on card 1): jump {t['jump'] / 1e6:.1f} MB, "
              f"CPU {t['cpu'] / 1e6:.1f} MB, one hand-off {handoff_tokens * n_tokens / 1e6:.1f} MB, "
              f"cards {u[0] / 2**30:.2f} + {u[1] / 2**30:.2f} GiB, one-card share {share:.1f}%")
        own = []
        for l in range(N_LAYER):
            c0 = int((labels_opt[l] == 0).sum())
            c1 = int((labels_opt[l] == 1).sum())
            own.append(0 if c0 > c1 else 1)
        switches = sum(1 for i in range(1, N_LAYER) if own[i] != own[i - 1]) + 1
        mv, nt = token_moves(labels, per_file)
        print(f"  contiguous, no home: {mv / max(nt, 1):.2f} jumps a token, {mv * handoff_tokens / 1e6:.1f} MB")
        print(f"  the free assignment changes card {switches} times a pass: "
              f"{switches * handoff_tokens * n_tokens / 1e6:.1f} MB of hand-offs, {switches} synchronisations a token")

    if a.home != "roam":
        home = int(a.home)
        labels_home = local_search(greedy_fill(events, blobs, caps, home), events, blobs, caps, k, jump_unit,
                                   cpu_unit, a.jump_per_expert, home, sweeps=4)
        t, per_dom = totals(labels_home, events, blobs, k, jump_unit, cpu_unit, a.jump_per_expert, domains, home)
        u = used_bytes(labels_home, blobs)
        share = (t["events"] - t["jumps"]) / t["events"] * 100
        print("")
        print(f"home card {home}, the other card computes its experts' rows: jump {t['jump'] / 1e6:.1f} MB over "
              f"{t['jumps']} of {t['events']} events ({share:.1f}% stay on the home card), CPU {t['cpu'] / 1e6:.1f} MB, "
              f"cards {u[0] / 2**30:.2f} + {u[1] / 2**30:.2f} GiB")

    rows = {}
    print("")
    print("| strategy | jump MB | jump events | one-card share | CPU MB | CPU lookups | card0 GiB | card1 GiB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, labels in (("rank", labels_rank), ("co-act", labels_opt), ("layer", labels_layer)):
        t, per_dom = totals(labels, events, blobs, k, jump_unit, cpu_unit, a.jump_per_expert, domains)
        u = used_bytes(labels, blobs)
        rows[name] = (t, per_dom, u)
        share = (t["events"] - t["jumps"]) / t["events"] * 100
        print(f"| {name} | {t['jump'] / 1e6:.1f} | {t['jumps']} | {share:.1f}% | {t['cpu'] / 1e6:.1f} | "
              f"{t['cpu_lookups']} | {u[0] / 2**30:.2f} | {u[1] / 2**30:.2f} |")
    r, o, L = rows["rank"][0], rows["co-act"][0], rows["layer"][0]
    print("")
    print(f"same experts on the cards: co-activation {o['jump'] / 1e6:.1f} MB against rank {r['jump'] / 1e6:.1f} MB "
          f"({(o['jump'] / max(r['jump'], 1) - 1) * 100:+.1f}%), events {o['jumps']} against {r['jumps']}")
    print(f"whole layers: {L['jump'] / 1e6:.1f} MB of jumps, CPU {L['cpu'] / 1e6:.1f} MB against {o['cpu'] / 1e6:.1f} MB")
    print("")
    for name in ("co-act", "layer"):
        for d, e in sorted(rows[name][1].items()):
            nm = dom_names[d] if 0 <= d < len(dom_names) else "?"
            print(f"  {name} / {nm}: {e['jumps']}/{e['events']} events jump "
                  f"({100 * e['jumps'] / max(e['events'], 1):.1f}%), CPU {e['cpu'] / 1e6:.1f} MB")

    if a.path_cost:
        tok = token_matrix(per_file)
        row = (5 * N_EMBD + 4) * dtype_bytes
        print("")
        print("a token's path - stay and fetch, or jump and continue:")
        print("| layout | bytes MB | jumps | CPU MB |")
        print("| --- | ---: | ---: | ---: |")
        for name, lab in (("rank", labels_rank), ("co-act", labels_opt), ("layer", labels_layer),
                          ("hot fill", greedy_fill(events, blobs, caps, 0))):
            b, jn, cpu = path_cost(lab, tok, blobs, row)
            print(f"| {name} | {b / 1e6:.1f} | {jn} | {cpu / 1e6:.1f} |")

    if a.duplicate_sweep:
        tok = token_matrix(per_file)
        row = (5 * N_EMBD + 4) * dtype_bytes
        print("")
        print("| duplicated on both cards | bytes MB | jumps | jumps a token | CPU MB |")
        print("| ---: | ---: | ---: | ---: | ---: |")
        for nd in [int(x) for x in a.duplicate_sweep.split(",") if x.strip()]:
            lab = fill_with_dupes(events, blobs, caps, nd)
            b, jn, cpu = path_cost(lab, tok, blobs, row)
            print(f"| {nd} | {b / 1e6:.1f} | {jn} | {jn / len(tok):.2f} | {cpu / 1e6:.1f} |")

    if a.optimize_jumps:
        tok = token_matrix(per_file)
        row_bytes = (5 * N_EMBD + 4) * dtype_bytes
        start = greedy_fill(events, blobs, caps, 0)
        j0 = jump_cost(start, tok, blobs, row_bytes)
        print("")
        print(f"jump search from the frequency fill: {j0[2]} jumps ({j0[2] / len(tok):.2f} a token), "
              f"{j0[0] / 1e6:.1f} MB, CPU {j0[1] / 1e6:.1f} MB")
        opt, jb = optimize_jumps(start, tok, blobs, caps, row_bytes, top=a.top)
        u = used_bytes(opt, blobs)
        print(f"after the search: {jb[2]} jumps ({jb[2] / len(tok):.2f} a token), {jb[0] / 1e6:.1f} MB, "
              f"CPU {jb[1] / 1e6:.1f} MB, cards {u[0] / 2**30:.2f} + {u[1] / 2**30:.2f} GiB")

    print("")
    print("with no home - the token continues on the card it jumps to:")
    row = (5 * N_EMBD + 4) * dtype_bytes
    for name, labels in (("rank", labels_rank), ("co-act", labels_opt), ("layer", labels_layer)):
        mv, nt = token_moves(labels, per_file)
        print(f"  {name}: {mv / max(nt, 1):.2f} jumps a token, {mv / 1e6:.0f}k rows, {mv * row / 1e6:.1f} MB")

    if a.out:
        np.savez_compressed(a.out, blobs=blobs, rank=labels_rank, coact=labels_opt, layer=labels_layer)
        print(f"\npartitions -> {a.out}")


if __name__ == "__main__":
    main()
