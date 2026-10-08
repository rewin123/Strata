#!/bin/bash
# PR 1401 confirmation on a V100 with v0.1.41: kernel checks with and without STRATA_SM70_TABLE=1, then the engine A/B
cd /root/strata
R=/root/results; mkdir -p $R
B=build-dev
M=/root/Strata-data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
{ git log --oneline -1; nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -1; lscpu | grep "Model name"; cat /sys/fs/cgroup/cpu.max; free -g | head -2; } > $R/00-rig.txt 2>&1
run() { name=$1; shift; echo "### $name: $*" ; ( "$@" ) 2>&1; echo "exit $?"; }
{
for e in 0 1; do
  run "native_grouped_parity SM70=$e" env STRATA_SM70_TABLE=$e $B/native_grouped_parity --bench
  run "mmvq_il_parity SM70=$e" env STRATA_SM70_TABLE=$e $B/mmvq_il_parity
  run "mmvq_multi_parity SM70=$e" env STRATA_SM70_TABLE=$e $B/mmvq_multi_parity
  run "gr_multi_parity SM70=$e" env STRATA_SM70_TABLE=$e $B/gr_multi_parity
  run "gr_parity SM70=$e" env STRATA_SM70_TABLE=$e $B/gr_parity
  run "fused_gr_bench SM70=$e" env STRATA_SM70_TABLE=$e $B/fused_gr_bench 300 1 4
  run "fused_gr_bench staged SM70=$e" env STRATA_SM70_TABLE=$e STRATA_HC_SPLIT=2 $B/fused_gr_bench 300 1 4
done
} > $R/10-kernel-checks.txt
run "mmvq_il_parity --bench SM70=1" env STRATA_SM70_TABLE=1 $B/mmvq_il_parity --bench > $R/20-mmvq-il-bench.txt
{
for T in 1 2 3 4; do run "native_expert_bench T=$T" env STRATA_SM70_TABLE=1 $B/native_expert_bench $M $(seq -s, 0 47) 28 $T 0 8 50; done
} > $R/30-native-expert-bench.txt
rm -f /tmp/ab_*_r*.log
STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 .venv/bin/python tools/ab_engine.py strata-iq2_xs.json \
  "off:exe=/root/strata/engine-cuda12/strata,xargs=--pool-workers;14" \
  "on:exe=/root/strata/engine-cuda12/strata,STRATA_SM70_TABLE=1,xargs=--pool-workers;14" \
  --rounds 3 --log $R/40-ab.jsonl > $R/40-ab.txt 2>&1
for f in /tmp/ab_*_r*.log; do echo "$(basename $f) $(grep -E 'decode GPU stages' $f | tail -1)"; echo "$(basename $f) $(grep -E 'decode timing' $f | tail -1)"; done > $R/41-ab-stage-lines.txt
cp /tmp/ab_*_r*.log $R/ 2>/dev/null
echo ALL_DONE > $R/99-done.txt
