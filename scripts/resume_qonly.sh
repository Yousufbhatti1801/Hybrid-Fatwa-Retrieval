#!/bin/zsh
set -euo pipefail

REPO="/Users/aaple/Documents/untitled folder/hybridretrieval+vectorless-rag/repos/Hybrid-Fatwa-Retrieval"
PY="/Users/aaple/Documents/untitled folder/hybridretrieval+vectorless-rag/.venv311/bin/python"

cd "$REPO"

# IMPORTANT: Do not delete .pipeline_cache_qonly if you want resume behavior.
nohup "$PY" orchestrator.py \
  --log-level INFO \
  --summary-json pipeline_summary_qonly.json \
  --work-dir .pipeline_cache_qonly \
  --batch-size 100 \
  >> pipeline_qonly.log 2>&1 &

echo "Started PID: $!"
echo "Resuming from existing checkpoint at .pipeline_cache_qonly/embed_checkpoint.db"
