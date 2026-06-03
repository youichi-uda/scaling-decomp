#!/bin/bash
# Phase 1: Cerebras-GPT N-axis decomposition (7 sizes, finals only)
# Phase 2: lm-eval on Pythia 2.8B/6.9B/12B + Cerebras-GPT 2.7B/6.7B/13B for downstream corr
set -uo pipefail
cd "$(dirname "$0")"

IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY=~/.runpod/ssh/runpodctl-ssh-key
PUBKEY="$(cat ${KEY}.pub)"
LOG=runpod_cerebras_eval.log

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
OUT=$(runpodctl pod create --name "sdcer-$(date +%H%M%S)" \
    --gpu-id "$GPU" --cloud-type SECURE --image "$IMAGE" \
    --container-disk-in-gb 200 --ports "22/tcp" --ssh --env "$ENV_JSON" 2>&1)
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

SSHO="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p $SSH_PORT"
SCPO="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $SSH_PORT"

for i in $(seq 1 60); do $SSHO root@$SSH_IP "echo OK" 2>/dev/null | grep -q OK && break; sleep 5; done
$SSHO root@$SSH_IP "echo OK" 2>&1 | grep -q OK || exit 4

$SCPO measure_v2.py root@$SSH_IP:/workspace/ 2>&1 | tail -1 | tee -a "$LOG"
$SSHO root@$SSH_IP "cd /workspace && pip install -q 'transformers==4.46.3' 'datasets==3.0.0' 'numpy<2' 'huggingface_hub[hf_transfer]' lm-eval 2>&1 | tail -3 && python -c 'from transformers import AutoModelForCausalLM; print(\"IMPORT OK\")' && python -c 'import lm_eval; print(\"lm-eval\", lm_eval.__version__)'" 2>&1 | tee -a "$LOG"

# Phase 1: Cerebras-GPT 7 sizes (finals only, N-axis decomp)
for SZ in "111M" "256M" "590M" "1.3B" "2.7B" "6.7B" "13B"; do
    BS=8
    [ "$SZ" = "2.7B" ] && BS=4
    [ "$SZ" = "6.7B" ] && BS=2
    [ "$SZ" = "13B" ] && BS=2
    OUTF="v2mt_cerebras-${SZ}.json"
    echo "[main] === cerebras-${SZ} (bs=$BS) ===" | tee -a "$LOG"
    $SSHO root@$SSH_IP "cd /workspace && HF_HUB_ENABLE_HF_TRANSFER=1 timeout 1800 python measure_v2.py \
        --model cerebras/Cerebras-GPT-${SZ} --docs 200 --seqlen 1024 --bs $BS --out $OUTF 2>&1 | tail -5" 2>&1 | tee -a "$LOG"
    $SCPO root@$SSH_IP:/workspace/$OUTF . 2>&1 | tail -1 | tee -a "$LOG"
    # Free disk
    $SSHO root@$SSH_IP "find /root/.cache/huggingface/hub/models--cerebras--Cerebras-GPT-${SZ}/blobs/ -size +100M -delete 2>/dev/null; df -h /root | tail -1" 2>&1 | tail -1 | tee -a "$LOG"
done

# Phase 2: lm-eval on Pythia 2.8B/6.9B/12B (small N done locally separately)
for M in EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b EleutherAI/pythia-12b; do
    SAFE=$(echo "$M" | tr '/' '_')
    OUTJ="lmeval_${SAFE}.json"
    echo "[main] === lm-eval $M ===" | tee -a "$LOG"
    $SSHO root@$SSH_IP "cd /workspace && HF_HUB_ENABLE_HF_TRANSFER=1 timeout 3600 lm-eval \
        --model hf --model_args pretrained=$M,dtype=bfloat16 \
        --tasks lambada_openai,piqa,hellaswag,arc_easy,sciq \
        --batch_size auto --output_path $OUTJ 2>&1 | tail -20" 2>&1 | tee -a "$LOG"
    $SCPO root@$SSH_IP:/workspace/$OUTJ . 2>&1 | tail -1 | tee -a "$LOG" || true
    $SCPO -r root@$SSH_IP:/workspace/$OUTJ . 2>&1 | tail -3 | tee -a "$LOG" || true
done

echo "[main] all done $(date)" | tee -a "$LOG"
