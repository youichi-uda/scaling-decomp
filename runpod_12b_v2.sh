#!/bin/bash
# Retry 12B run on RunPod with longer SSH wait + COMMUNITY first, then SECURE A100.
set -uo pipefail
cd "$(dirname "$0")"

IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY=~/.runpod/ssh/runpodctl-ssh-key
PUBKEY="$(cat ${KEY}.pub)"
LOG=runpod_12b_v2.log

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

try_launch() {
    local GPU="$1" CLOUD="$2"
    echo "[try] $GPU $CLOUD" | tee -a "$LOG"
    ENV_JSON=$(printf '{"PUBLIC_KEY":"%s"}' "$PUBKEY")
    OUT=$(runpodctl pod create \
        --name "sd12b-$(date +%H%M%S)" \
        --gpu-id "$GPU" \
        --cloud-type "$CLOUD" \
        --image "$IMAGE" \
        --container-disk-in-gb 80 \
        --ports "22/tcp" \
        --ssh \
        --env "$ENV_JSON" 2>&1)
    echo "$OUT" >> "$LOG"
    local PID=$(echo "$OUT" | ~/.venv-bigshrink312/bin/python -c "
import sys, json, re
txt=sys.stdin.read()
m=re.search(r'\{.*\}', txt, re.S)
if m:
  try: d=json.loads(m.group(0)); print(d.get('id',''))
  except: pass" 2>/dev/null)
    if [ -z "$PID" ]; then
        echo "[try] $GPU $CLOUD: no POD_ID returned" | tee -a "$LOG"
        return 1
    fi
    POD_ID="$PID"
    echo "[try] POD_ID=$POD_ID" | tee -a "$LOG"

    # Wait up to 12 min for SSH endpoint
    SSH_IP=""; SSH_PORT=""
    for i in $(seq 1 144); do
        INFO=$(runpodctl ssh info "$POD_ID" 2>&1 | ~/.venv-bigshrink312/bin/python -c "
import sys, json, re
txt=sys.stdin.read()
m=re.search(r'\{.*\}', txt, re.S)
if m:
  try:
    d=json.loads(m.group(0))
    ip=d.get('ip',''); port=d.get('port','')
    if ip and port and str(port)!='0': print(f'{ip}|{port}')
  except: pass")
        if [ -n "$INFO" ]; then
            SSH_IP="${INFO%|*}"; SSH_PORT="${INFO#*|}"
            echo "[try] ssh ip=$SSH_IP port=$SSH_PORT (after $((i*5))s)" | tee -a "$LOG"
            break
        fi
        sleep 5
    done
    if [ -z "$SSH_IP" ]; then
        echo "[try] $GPU $CLOUD: no SSH endpoint after 12 min, abort" | tee -a "$LOG"
        runpodctl pod remove "$POD_ID" 2>&1 | tail -2 >> "$LOG" || true
        POD_ID=""
        return 1
    fi

    SSHO="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p $SSH_PORT"
    SCPO="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $SSH_PORT"
    for i in $(seq 1 60); do
        $SSHO root@$SSH_IP "echo OK" 2>/dev/null | grep -q OK && { echo "[try] sshd OK"; break; }
        sleep 5
    done
    $SSHO root@$SSH_IP "echo OK" 2>&1 | grep -q OK || {
        echo "[try] sshd never accepted" | tee -a "$LOG"
        return 1
    }

    $SCPO measure_v2.py root@$SSH_IP:/workspace/ 2>&1 | tail -2 | tee -a "$LOG"
    $SSHO root@$SSH_IP "cd /workspace && \
        pip install -q 'transformers==4.46.3' 'datasets==3.0.0' 'numpy<2' 2>&1 | tail -3 && \
        python -c 'from transformers import AutoModelForCausalLM, AutoTokenizer; print(\"IMPORT OK\")' && \
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
        python measure_v2.py --model EleutherAI/pythia-12b --docs 200 --seqlen 1024 --bs 4 --out v2_12b.json 2>&1 | tail -20 && \
        ls -la v2_12b.json && \
        python -c 'import json; d=json.load(open(\"v2_12b.json\")); print(\"VERIFIED overall=\", d[\"overall_mean_loss\"])'" 2>&1 | tee -a "$LOG"

    $SCPO root@$SSH_IP:/workspace/v2_12b.json . 2>&1 | tail -2 | tee -a "$LOG"
    if [ -f v2_12b.json ]; then
        ~/.venv-bigshrink312/bin/python -c "import json; d=json.load(open('v2_12b.json')); print('LOCAL VERIFIED', d['model'], 'overall=', d['overall_mean_loss'])" | tee -a "$LOG" && return 0
    fi
    echo "[try] no v2_12b.json after scp" | tee -a "$LOG"
    return 1
}

: > "$LOG"
echo "[main] start $(date)" | tee -a "$LOG"
# H100 SECURE worked for SSH endpoint last time (failed only on transformers ABI, now fixed)
for GPU_CLOUD in "NVIDIA H100 80GB HBM3|SECURE"; do
    GPU="${GPU_CLOUD%|*}"; CLOUD="${GPU_CLOUD#*|}"
    POD_ID=""
    if try_launch "$GPU" "$CLOUD"; then
        echo "[main] SUCCESS with $GPU $CLOUD" | tee -a "$LOG"
        exit 0
    fi
    echo "[main] failed $GPU $CLOUD, next..." | tee -a "$LOG"
done
echo "[main] all GPU options exhausted" | tee -a "$LOG"
exit 7
