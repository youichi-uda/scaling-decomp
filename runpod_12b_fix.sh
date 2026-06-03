#!/bin/bash
# Re-run pythia-12b step50000 with use_safetensors=False (HF Hub bug fix).
set -uo pipefail
cd "$(dirname "$0")"

IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY=~/.runpod/ssh/runpodctl-ssh-key
PUBKEY="$(cat ${KEY}.pub)"
LOG=runpod_12b_fix.log

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
GPU="NVIDIA H100 80GB HBM3"
echo "[main] start $(date)" | tee -a "$LOG"
ENV_JSON=$(printf '{"PUBLIC_KEY":"%s"}' "$PUBKEY")
OUT=$(runpodctl pod create --name "sd12fix-$(date +%H%M%S)" \
    --gpu-id "$GPU" --cloud-type SECURE --image "$IMAGE" \
    --container-disk-in-gb 100 --ports "22/tcp" --ssh --env "$ENV_JSON" 2>&1)
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
$SSHO root@$SSH_IP "echo OK" 2>&1 | grep -q OK || exit 4

$SCPO measure_v2.py root@$SSH_IP:/workspace/ 2>&1 | tail -1 | tee -a "$LOG"
$SSHO root@$SSH_IP "cd /workspace && pip install -q 'transformers==4.46.3' 'datasets==3.0.0' 'numpy<2' 'huggingface_hub[hf_transfer]' 2>&1 | tail -2 && python -c 'from transformers import AutoModelForCausalLM; print(\"OK\")'" 2>&1 | tee -a "$LOG"

echo "[main] === 12b step50000 (use_safetensors=False auto via measure_v2) ===" | tee -a "$LOG"
$SSHO root@$SSH_IP "cd /workspace && HF_HUB_ENABLE_HF_TRANSFER=1 timeout 1800 python measure_v2.py \
    --model EleutherAI/pythia-12b --revision step50000 \
    --docs 200 --seqlen 1024 --bs 2 --out v2mt_12b_step50000.json 2>&1 | tail -8" 2>&1 | tee -a "$LOG"
$SCPO root@$SSH_IP:/workspace/v2mt_12b_step50000.json . 2>&1 | tail -1 | tee -a "$LOG"

# Also check all other 12b dense ckpts for the safetensors bug (early ones might be affected)
for S in step10000 step15000 step22000 step25000 step30000 step42000 step70000 step100000 step130000; do
    echo "[main] === 12b $S verify ===" | tee -a "$LOG"
    $SSHO root@$SSH_IP "cd /workspace && HF_HUB_ENABLE_HF_TRANSFER=1 timeout 1800 python measure_v2.py \
        --model EleutherAI/pythia-12b --revision $S \
        --docs 200 --seqlen 1024 --bs 2 --out v2mt_12b_${S}.json 2>&1 | tail -3" 2>&1 | tee -a "$LOG"
    $SCPO root@$SSH_IP:/workspace/v2mt_12b_${S}.json . 2>&1 | tail -1 | tee -a "$LOG"
done

echo "[main] done $(date)" | tee -a "$LOG"
