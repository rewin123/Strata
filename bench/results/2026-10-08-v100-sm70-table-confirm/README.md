# PR 1401 on a V100 with v0.1.41: `STRATA_SM70_TABLE=1` confirmed (2026-10-08)

The checks the maintainers asked for, run on the final code: tag `v0.1.41` (`fb58e0d`), nothing changed. Everything
below is bitwise where it says so; the speed numbers are this rig's.

## Rig

- Tesla V100-SXM2-32GB (sm_70, PCIe gen3 x16), driver 580.159.03, max SM clock 1530 MHz; a rented container (vast.ai):
  2x Xeon Gold 6130, a cgroup quota of ~15 CPUs (`cpu.max 1536000 100000`), 376 GB RAM. CUDA 12.8, GCC 11.4.
- Setup's own install, no edits: `./setup.sh --cuda 12 --build --model IQ2_XS --gpu 0 --vision no --yes` (its config:
  `--spec 4 --kv int8 --kv-resident 32768 --max-context 131072 --expert-cache auto --prefill auto`). The engine of both
  A/B arms is setup's `engine-cuda12/strata`; the kernel tests are a `-DSTRATA_BUILD_TESTS=ON` build of the same tag
  (`-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`).
- The script that ran all of it: `raw/run_all.sh`.

## Kernel checks (`raw/batch1/10-kernel-checks.txt`)

Each run with `STRATA_SM70_TABLE=0` and with `=1`:

| Check | `=0` | `=1` |
| --- | --- | --- |
| `native_grouped_parity --bench` | 0 failures | 0 failures |
| `mmvq_il_parity` | OK (every case, T 2-4, rows 1/2/4 and the table, bitwise) | OK (same) |
| `mmvq_multi_parity` | ok | ok |
| `gr_multi_parity` | OK (32 cases) | OK (32 cases) |
| `gr_parity` | 0 failures | 0 failures |
| `fused_gr_bench 300 1 4` (plain; and with `STRATA_HC_SPLIT=2`, the engine's staged read) | 0 differing | 0 differing |

**`native_expert_bench`** on the IQ2_XS shard, all 48 layers (34 IQ2_S / 11 IQ2_XXS / 3 IQ1_M gate/up, Q2_0 down),
28 groups, T = 1-4, mode 0 vs mode 8 (`raw/batch1/30-native-expert-bench.txt`): **192 of 192 bitwise equal.** Summed
over the 48 layers (one verify window's VRAM calls):

| T | mode 0 | mode 8 | |
| ---: | ---: | ---: | ---: |
| 1 | 5.68 ms | 4.90 ms | -13.8% |
| 2 | 7.02 ms | 5.92 ms | -15.6% |
| 3 | 8.53 ms | 7.07 ms | -17.1% |
| 4 | 9.92 ms | 8.40 ms | -15.3% |

**`mmvq_il_parity --bench`** with `=1` (`raw/batch1/20-mmvq-il-bench.txt`): the same as the bench the table was read
from, e.g. at 3 columns the Q5_K head 1,108.9 -> 838.3 us (rows 2), Q6_K 12,288 rows 70.5 -> 50.6 us, Q4_K 10,240 rows
44.3 -> 32.5 us, IQ4_XS 6,144 rows 28.3 -> 19.2 us.

**`fused_gr_bench`**, the staged read the engine runs on this card, old -> fast up: T 1 34.7 -> 33.2 us, T 4 50.6 ->
48.0 us (bitwise).

## Engine A/B (`tools/ab_engine.py`)

`off`: setup's engine as is (sm_70 as in 0.1.40.3). `on`: the same binary with `STRATA_SM70_TABLE=1`. Both with
`--pool-workers 14` (the container's quota; with setup's 32 workers the pool is throttled), `STRATA_DECODE_TIMING=1
STRATA_VERIFY_PROFILE=1`. Two batches, 3 + 5 rounds, the arms' order alternating, one server start per arm and round;
per round three chats (256 tokens) and a 6,345-token prompt (122-132 tokens). Raw: `raw/batch1/`, `raw/batch2/`
(`40-ab.txt`, `40-ab.jsonl`, `41-ab-stage-lines.txt` and every server log).

**GPU work per verify window** (the engine's stage table, its waits `waitA` / `waitB` / `waitCPU` left out) - every
`on` run is below every `off` run:

| | off (8 runs) | on (8 runs) |
| --- | --- | --- |
| range | 22.11 - 22.83 ms | 21.00 - 21.37 ms |
| median | 22.6 ms | 21.1 ms (-6.4%) |

Stage medians of the first batch (ms per window): routed experts in VRAM 4.30 -> 3.71, q8 + qkv GEMV 1.54 -> 1.39,
z 1.01 -> 0.88, head 0.95 -> 0.70, hc-read0 up 1.04 -> 0.97, q + q-idx 0.73 -> 0.67; out-proj 1.32 -> 1.39 (the one
stage that moves the other way); the rest unchanged.

**Decode tokens/s.** On this rig decode is dominated by the CPU expert pool: in 10 of the 16 server starts the
windows' `waitCPU` was 8-49 ms (whatever the arm; one `on` start fell to 8.5 tok/s), in the other 6 it was 0.5 ms.
Those 6 - three per arm - are the comparison; their spread is under 2%:

| Request | off (3 clean runs) | on (3 clean runs) | |
| --- | --- | --- | ---: |
| chat 0 | 87.7 / 89.0 / 88.2 | 92.8 / 92.5 / 92.4 | +4.9% |
| chat 1 | 69.7 / 71.0 / 70.5 | 74.6 / 75.6 / 75.3 | +6.8% |
| chat 2 | 67.3 / 68.6 / 69.3 | 73.2 / 73.5 / 73.1 | +6.7% |
| 6,345-token prompt | 62.0 / 62.5 / 62.6 | 66.2 / 66.0 / 65.8 | +5.6% |

Medians over all 8 rounds, noise included: chat 0 62.8 / 57.7, chat 1 57.8 / 58.2, chat 2 59.5 / 59.8, long 48.5 /
50.9. Prompt speed does not change (1,324 / 1,344 tok/s, medians of the 6,345-token prompt): the prompt path uses none
of these kernels.

**Answers.** The first 80 characters of every answer are the same in both arms (one text per chat request, two for
the long prompt, in both arms). Answer lengths vary between starts within each arm as well (chat 1: 216-256 tokens
off, 216-256 on), as the CPU experts' share of the window changes from start to start.

## One more finding (not in 0.1.41, a suggestion)

On CUDA a format without a shared-memory kernel falls back to the CUDA layout (`kExpFallback = 0`, chosen from IQ2_XXS
**gate/up**, where R2 was slower: 183 vs 113 us). For the **down** projection that is not the right choice everywhere:
the synthetic window of `native_grouped_parity --bench` (IQ3_S gate/up, IQ4_XS down - UD-IQ4_XS's formats), the
VRAM call with `=1`:

| down IQ4_XS takes | VRAM call (28 groups, 40 entries) |
| --- | ---: |
| CUDA layout (0.1.41) | 196.2 us |
| R2 (a temporary `STRATA_EXP_DOWN_FALLBACK=2` patch, not committed) | 159.8 us |
| `=0` (mode 0 everywhere), for reference | 229.0 us |

`native_grouped_parity` passes with R2 there, but it compares each mode with itself; whether R2's IQ4_XS down is bitwise
the CUDA layout's (needed, as the window's PCIe call keeps the CUDA layout) was not checked - that needs
`native_expert_bench` on the UD-IQ4_XS shard, which this rig did not have. The IQ2_XS pack is not affected (its Q2_0
down has a shared-memory kernel).
