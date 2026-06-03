#!/bin/bash
# Run measure_v2.py on pythia-12b via RunPod H100 80GB.
# Trap-deletes pod on any exit. Verifies result on disk before exiting OK.
set -uo pipefail
cd "$(dirname "$0")"

GPU="NVIDIA H100 80GB HBM3"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY=~/.runpod/ssh/runpodctl-ssh-key
PUBKEY="$(cat ${KEY}.pub)"
LOG=runpod_12b.log

POD_ID=""
cleanup() {
    if [ -n "$POD_ID" ]; then
        echo "[cleanup] deleting pod $POD_ID" | tee -a "$LOG"
        runpodctl pod stop "$POD_ID" 2>&1 | tail -2 >> "$LOG"
        runpodctl pod remove "$POD_ID" 2>&1 | tail -2 >> "$LOG"
        sleep 3
        if runpodctl pod list 2>/dev/null | grep -q "\"$POD_ID\""; then
            echo "[cleanup] retry remove" | tee -a "$LOG"
            runpodctl pod remove "$POD_ID" 2>&1 | tail -2 >> "$LOG"
        fi
        echo "[cleanup] done" | tee -a "$LOG"
    fi
}
trap cleanup EXIT INT TERM

: > "$LOG"
echo "[launch] $GPU SECURE" | tee -a "$LOG"
ENV_JSON=$(printf '{"PUBLIC_KEY":"%s"}' "$PUBKEY")
CREATE_OUT=$(runpodctl pod create \
    --name "scaling-decomp-12b" \
    --gpu-id "$GPU" \
    --cloud-type SECURE \
    --image "$IMAGE" \
    --container-disk-in-gb 80 \
    --ports "22/tcp" \
    --ssh \
    --env "$ENV_JSON" 2>&1)
echo "$CREATE_OUT" | tee -a "$LOG"
POD_ID=$(echo "$CREATE_OUT" | grep -oE 'pod "[a-z0-9]+" created' | grep -oE '"[a-z0-9]+"' | tr -d '"' | head -1)
if [ -z "$POD_ID" ]; then
    POD_ID=$(echo "$CREATE_OUT" | grep -oE '"id"\s*:\s*"[^"]+"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
fi
echo "[launch] POD_ID=$POD_ID" | tee -a "$LOG"
[ -z "$POD_ID" ] && { echo "[fatal] no POD_ID"; exit 2; }

# Poll for ssh endpoint
echo "[wait] ssh endpoint..." | tee -a "$LOG"
SSH_IP=""; SSH_PORT=""
for i in $(seq 1 60); do
    INFO=$(runpodctl ssh info "$POD_ID" 2>&1 | grep -oE '\{.*\}' | head -1)
    if [ -n "$INFO" ]; then
        SSH_IP=$(echo "$INFO" | ~/.venv-bigshrink312/bin/python -c "import sys,json; d=json.load(sys.stdin); print(d.get('ip',''))" 2>/dev/null)
        SSH_PORT=$(echo "$INFO" | ~/.venv-bigshrink312/bin/python -c "import sys,json; d=json.load(sys.stdin); print(d.get('port',''))" 2>/dev/null)
        if [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ] && [ "$SSH_PORT" != "0" ]; then
            echo "[wait] ssh ip=$SSH_IP port=$SSH_PORT" | tee -a "$LOG"
            break
        fi
    fi
    sleep 5
done
[ -z "$SSH_IP" ] && { echo "[fatal] no SSH endpoint after 5 min"; exit 3; }

SSHO="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p $SSH_PORT"
SCPO="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $SSH_PORT"

# Wait for ssh to actually accept
echo "[wait] sshd accept..." | tee -a "$LOG"
for i in $(seq 1 60); do
    $SSHO root@$SSH_IP "echo OK" 2>/dev/null | grep -q OK && { echo "[wait] sshd OK"; break; }
    sleep 5
done
$SSHO root@$SSH_IP "echo OK" 2>&1 | grep -q OK || { echo "[fatal] sshd never accepted"; exit 4; }

# Copy script
$SCPO measure_v2.py root@$SSH_IP:/workspace/ 2>&1 | tail -2 | tee -a "$LOG"

# Install deps + verify import + run
$SSHO root@$SSH_IP "cd /workspace && \
    pip install -q 'transformers==5.9.0' 'datasets==4.8.5' 2>&1 | tail -5 && \
    python -c 'from transformers import AutoModelForCausalLM, AutoTokenizer; print(\"IMPORT OK\")' && \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
    python measure_v2.py --model EleutherAI/pythia-12b --docs 200 --seqlen 1024 --bs 2 --out v2_12b.json 2>&1 | tail -30 && \
    ls -la v2_12b.json && \
    python -c 'import json; d=json.load(open(\"v2_12b.json\")); print(\"VERIFIED overall=\", d[\"overall_mean_loss\"])'" 2>&1 | tee -a "$LOG"

# Pull result
$SCPO root@$SSH_IP:/workspace/v2_12b.json . 2>&1 | tail -2 | tee -a "$LOG"

# VERIFY on local disk before declaring success
if [ ! -f v2_12b.json ]; then
    echo "[fatal] v2_12b.json missing locally after scp"; exit 5
fi
~/.venv-bigshrink312/bin/python -c "import json; d=json.load(open('v2_12b.json')); print('LOCAL VERIFIED', d['model'], 'overall=', d['overall_mean_loss'])" | tee -a "$LOG" || { echo "[fatal] local JSON broken"; exit 6; }

echo "[done] v2_12b.json verified locally" | tee -a "$LOG"
