#!/bin/bash
# 6.9B only D-axis run. Per-step timeout to avoid HF Hub download stalls.
set -uo pipefail
cd "$(dirname "$0")"

IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY=~/.runpod/ssh/runpodctl-ssh-key
PUBKEY="$(cat ${KEY}.pub)"
LOG=runpod_6.9b_only.log

POD_ID=""
cleanup() {
    if [ -n "$POD_ID" ]; then
        echo "[cleanup] $POD_ID" | tee -a "$LOG"
        runpodctl pod stop "$POD_ID" 2>&1 | tail -2 >> "$LOG" || true
        sleep 2
        runpodctl pod remove "$POD_ID" 2>&1 | tail -2 >> "$LOG" || true
        sleep 3
        if runpodctl pod list 2>/dev/null | grep -q "\"$POD_ID\""; then
            runpodctl pod remove "$POD_ID" 2>&1 | tail -2 >> "$LOG" || true
        fi
    fi
}
trap cleanup EXIT INT TERM

: > "$LOG"
GPU="NVIDIA H100 80GB HBM3"
echo "[main] $GPU SECURE start $(date)" | tee -a "$LOG"
ENV_JSON=$(printf '{"PUBLIC_KEY":"%s"}' "$PUBKEY")
OUT=$(runpodctl pod create \
    --name "sd69-$(date +%H%M%S)" \
    --gpu-id "$GPU" --cloud-type SECURE --image "$IMAGE" \
    --container-disk-in-gb 150 --ports "22/tcp" --ssh \
    --env "$ENV_JSON" 2>&1)
echo "$OUT" >> "$LOG"
POD_ID=$(echo "$OUT" | ~/.venv-bigshrink312/bin/python -c "
import sys, json, re
txt=sys.stdin.read(); m=re.search(r'\{.*\}', txt, re.S)
if m:
  try: d=json.loads(m.group(0)); print(d.get('id',''))
  except: pass" 2>/dev/null)
[ -z "$POD_ID" ] && { echo "[main] no POD_ID"; exit 2; }
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
        echo "[main] ssh $SSH_IP:$SSH_PORT after $((i*5))s" | tee -a "$LOG"; break; fi
    sleep 5
done
[ -z "$SSH_IP" ] && { echo "[main] no SSH"; exit 3; }

SSHO="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=30 -p $SSH_PORT"
SCPO="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $SSH_PORT"

for i in $(seq 1 60); do
    $SSHO root@$SSH_IP "echo OK" 2>/dev/null | grep -q OK && break
    sleep 5
done
$SSHO root@$SSH_IP "echo OK" 2>&1 | grep -q OK || { echo "[main] sshd nope"; exit 4; }

$SCPO measure_v2.py root@$SSH_IP:/workspace/ 2>&1 | tail -2 | tee -a "$LOG"
$SSHO root@$SSH_IP "cd /workspace && \
    pip install -q 'transformers==4.46.3' 'datasets==3.0.0' 'numpy<2' 'huggingface_hub[hf_transfer]' 2>&1 | tail -3 && \
    python -c 'from transformers import AutoModelForCausalLM; print(\"IMPORT OK\")' && \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>&1 | tee -a "$LOG"

# Run 4 D-axis checkpoints with 25-min per-step timeout (hf_transfer enabled for fast download)
for STEP in 10000 25000 50000 100000; do
    OUTF="v2mt_6.9b_step${STEP}.json"
    echo "[main] === 6.9b step=$STEP (timeout 25m) ===" | tee -a "$LOG"
    $SSHO root@$SSH_IP "cd /workspace && HF_HUB_ENABLE_HF_TRANSFER=1 timeout 1500 python measure_v2.py \
        --model EleutherAI/pythia-6.9b --revision step$STEP \
        --docs 200 --seqlen 1024 --bs 4 --out $OUTF 2>&1 | tail -10" 2>&1 | tee -a "$LOG"
    $SCPO root@$SSH_IP:/workspace/$OUTF . 2>&1 | tail -1 | tee -a "$LOG"
    if [ -f "$OUTF" ]; then
        ~/.venv-bigshrink312/bin/python -c "import json; d=json.load(open('$OUTF')); print('  LOCAL', d['model'], 'rev=step$STEP', 'overall=', d['overall_mean_loss'])" | tee -a "$LOG"
    else
        echo "[main] MISSING $OUTF (timed out or failed)" | tee -a "$LOG"
    fi
done
echo "[main] all 6.9b done $(date)" | tee -a "$LOG"
