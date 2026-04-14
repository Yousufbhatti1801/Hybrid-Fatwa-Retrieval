#!/bin/zsh
set -euo pipefail

REPO="/Users/aaple/Documents/untitled folder/hybridretrieval+vectorless-rag/repos/Hybrid-Fatwa-Retrieval"
PY="/Users/aaple/Documents/untitled folder/hybridretrieval+vectorless-rag/.venv311/bin/python"

cd "$REPO"

ps aux | grep "orchestrator.py" | grep -v grep || echo "No running orchestrator"

echo "--- tail pipeline_qonly.log ---"
tail -8 pipeline_qonly.log || true

echo "--- checkpoint count ---"
if [[ -f .pipeline_cache_qonly/embed_checkpoint.db ]]; then
  "$PY" -c "import sqlite3; c=sqlite3.connect('.pipeline_cache_qonly/embed_checkpoint.db'); print(c.execute('select count(*) from embeddings').fetchone()[0]); c.close()"
else
  echo "Checkpoint DB not found"
fi

echo "--- pinecone count ---"
"$PY" -c "from src.indexing.pinecone_store import init_index; s=init_index().describe_index_stats(); print(s.get('total_vector_count', 0))"
