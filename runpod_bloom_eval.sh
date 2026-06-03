#!/bin/bash
# BLOOM 4th-family replication + lm-eval Pythia 6.9B/12B retry with fresh install
set -uo pipefail
cd "$(dirname "$0")"

IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY=~/.runpod/ssh/runpodctl-ssh-key
PUBKEY="$(cat ${KEY}.pub)"
LOG=runpod_bloom_eval.log

POD_ID=""
cleanup() {
    if [ -n "$POD_ID" ]; then
        echo "[cleanup] $POD_ID" | tee -a "$LOG"
        runpodctl pod stop "$POD_ID" 2>&1 | tail -2 >> "$LOG" || true
        sleep 2
        runpodctl pod remove "$POD_ID" 2>&1 | tail -2 >> "$LOG" || true
        sleep 3
    fi
}
trap cleanup EXIT INT TERM

: > "$LOG"
echo "[main] start $(date)" | tee -a "$LOG"
ENV_JSON=$(printf '{"PUBLIC_KEY":"%s"}' "$PUBKEY")
OUT=$(runpodctl pod create --name "sdbloom-$(date +%H%M%S)" \
    --gpu-id "NVIDIA H100 80GB HBM3" --cloud-type SECURE --image "$IMAGE" \
    --container-disk-in-gb 250 --ports "22/tcp" --ssh --env "$ENV_JSON" 2>&1)
POD_ID=$(echo "$OUT" | ~/.venv-bigshrink312/bin/python -c "
import sys, json, re
txt=sys.stdin.read(); m=re.search(r'\{.*\}', txt, re.S)
if m:
  try: d=json.loads(m.group(0)); print(d.get('id',''))
  except: pass" 2>/dev/null)
echo "[main] POD_ID=$POD_ID" | tee -a "$LOG"

SSH_IP=""; SSH_PORT=""
for i in $(seq 1 144); do
    INFO=$(runpodctl ssh info "$POD_ID" 2>&1 | ~/.venv-bigshrink312/bin/python -c "
import sys, json, re
txt=sys.stdin.read(); m=re.search(r'\{.*\}', txt, re.S)
if m:
  try:
    d=json.loads(m.group(0)); ip=d.get('ip',''); port=d.get('port','')
    if ip and port and str(port)!='0': print(f'{ip}|{port}')
  except: pass")
    if [ -n "$INFO" ]; then SSH_IP="${INFO%|*}"; SSH_PORT="${INFO#*|}"
        echo "[main] ssh $SSH_IP:$SSH_PORT" | tee -a "$LOG"; break; fi
    sleep 5
done
[ -z "$SSH_IP" ] && exit 3

SSHO="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT"
SCPO="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $SSH_PORT"
for i in $(seq 1 60); do $SSHO root@$SSH_IP "echo OK" 2>/dev/null | grep -q OK && break; sleep 5; done

$SCPO measure_v2.py root@$SSH_IP:/workspace/ 2>&1 | tail -1 | tee -a "$LOG"
# Install with explicit version compat to avoid TypedDict bug
$SSHO root@$SSH_IP "cd /workspace && pip install -q 'transformers==4.46.3' 'datasets==3.0.0' 'numpy<2' 'huggingface_hub[hf_transfer]' 'accelerate' 'lm-eval==0.4.5' 2>&1 | tail -5 && python -c 'from transformers import AutoModelForCausalLM; print(\"OK\")' && python -c 'import lm_eval; print(\"lm-eval\", lm_eval.__version__)'" 2>&1 | tee -a "$LOG"

# Phase 1: BLOOM 5 sizes (4th family)
for SZ in "560m" "1b1" "1b7" "3b" "7b1"; do
    BS=8
    [ "$SZ" = "3b" ] && BS=4
    [ "$SZ" = "7b1" ] && BS=2
    OUTF="v2mt_bloom-${SZ}.json"
    echo "[main] === bloom-${SZ} ===" | tee -a "$LOG"
    $SSHO root@$SSH_IP "cd /workspace && HF_HUB_ENABLE_HF_TRANSFER=1 timeout 1800 python measure_v2.py \
        --model bigscience/bloom-${SZ} --docs 200 --seqlen 1024 --bs $BS --out $OUTF 2>&1 | tail -5" 2>&1 | tee -a "$LOG"
    $SCPO root@$SSH_IP:/workspace/$OUTF . 2>&1 | tail -1 | tee -a "$LOG"
    $SSHO root@$SSH_IP "find /root/.cache/huggingface/hub/models--bigscience--bloom-${SZ}/blobs/ -size +100M -delete 2>/dev/null; df -h /root | tail -1" 2>&1 | tail -1 | tee -a "$LOG"
done

# Phase 2: lm-eval Pythia 6.9B and 12B (retry)
for M in EleutherAI/pythia-6.9b EleutherAI/pythia-12b; do
    SAFE=$(echo "$M" | tr "/" "_")
    OUTJ="lmeval_${SAFE}"
    echo "[main] === lm-eval $M ===" | tee -a "$LOG"
    $SSHO root@$SSH_IP "cd /workspace && HF_HUB_ENABLE_HF_TRANSFER=1 timeout 3600 lm-eval \
        --model hf --model_args pretrained=$M,dtype=bfloat16 \
        --tasks lambada_openai,piqa,arc_easy,sciq \
        --batch_size auto --output_path $OUTJ 2>&1 | tail -10" 2>&1 | tee -a "$LOG"
    $SCPO -r root@$SSH_IP:/workspace/$OUTJ . 2>&1 | tail -2 | tee -a "$LOG" || true
done

echo "[main] all done $(date)" | tee -a "$LOG"
