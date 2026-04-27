#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

API_URL="${API_URL:-http://127.0.0.1:8000/recommend}"
IMAGE_DIR="${IMAGE_DIR:-$REPO_ROOT/example_photos}"
WARMUP_COUNT="${WARMUP_COUNT:-3}"
SLEEP_BETWEEN_SEC="${SLEEP_BETWEEN_SEC:-0.2}"
OUTPUT_FILE="${OUTPUT_FILE:-$REPO_ROOT/bench/benchmark_$(date +%Y%m%d_%H%M%S).jsonl}"
BENCH_API_KEY="${BENCH_API_KEY:-}"

mkdir -p "$(dirname "$OUTPUT_FILE")"
: > "$OUTPUT_FILE"

images=()
while IFS= read -r image_path; do
  images+=("$image_path")
done < <(find "$IMAGE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)

if [[ ${#images[@]} -eq 0 ]]; then
  echo "No images found in $IMAGE_DIR"
  exit 1
fi

request_code() {
  local output_file="$1"
  local image_path="$2"

  if [[ -n "$BENCH_API_KEY" ]]; then
    curl -sS -o "$output_file" -w "%{http_code}" -H "X-API-Key: $BENCH_API_KEY" -F "file=@$image_path" "$API_URL" || true
  else
    curl -sS -o "$output_file" -w "%{http_code}" -F "file=@$image_path" "$API_URL" || true
  fi
}

echo "Benchmark target: $API_URL"
echo "Image directory:  $IMAGE_DIR"
echo "Output file:      $OUTPUT_FILE"
echo "Warmup requests:  $WARMUP_COUNT"
echo "Image count:      ${#images[@]}"

first_image="${images[0]}"
for ((i = 1; i <= WARMUP_COUNT; i++)); do
  tmp_body="$(mktemp)"
  code="$(request_code "$tmp_body" "$first_image")"
  rm -f "$tmp_body"
  echo "Warmup $i/$WARMUP_COUNT -> HTTP $code"
done

for img in "${images[@]}"; do
  tmp_body="$(mktemp)"
  code="$(request_code "$tmp_body" "$img")"

  python3 - "$img" "$code" "$tmp_body" "$OUTPUT_FILE" <<'PY'
import json
import sys

img_path, code_raw, body_path, out_path = sys.argv[1:]

try:
    status = int(code_raw)
except ValueError:
    status = -1

with open(body_path, "r", encoding="utf-8", errors="replace") as f:
    body_text = f.read().strip()

try:
    body = json.loads(body_text) if body_text else {}
except json.JSONDecodeError:
    body = {"raw": body_text}

record = {
    "file": img_path,
    "status": status,
    "body": body,
}

with open(out_path, "a", encoding="utf-8") as out:
    out.write(json.dumps(record, ensure_ascii=True) + "\n")
PY

  rm -f "$tmp_body"
  echo "Recorded: $img -> HTTP $code"
  sleep "$SLEEP_BETWEEN_SEC"
done

echo ""
echo "Benchmark run complete."
python3 "$SCRIPT_DIR/summarize.py" "$OUTPUT_FILE"
