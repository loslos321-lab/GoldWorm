#!/usr/bin/env bash
# ============================================================================
#  download_datasets.sh — TinyStories + WikiText-103 for Deep-Sleep Marathon
# ============================================================================
#  Requirements: python3 + huggingface_hub (CLI), ~3 GB free disk
# ============================================================================
set -euo pipefail

DATA_DIR="training_data"
MIN_BYTES=$((10 * 1024 * 1024))   # 10 MB sanity floor

mkdir -p "$DATA_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Downloading Datasets for Deep-Sleep Mega-Marathon          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Prerequisites ──
if ! command -v python3 &>/dev/null; then
    echo "✗ python3 is required but not found. Install it first."
    exit 1
fi

echo "── Checking hf CLI ──"
if ! command -v hf &>/dev/null; then
    echo "  hf CLI not found — installing..."
    pip3 install huggingface-hub -q
    echo "  ✓ installed."
else
    echo "  ✓ hf CLI available."
fi
echo ""

# ============================================================================
# 1. TinyStories — syntactic grammar structure (~1.5 GB)
# ============================================================================
TARGET_TS="$DATA_DIR/TinyStories.txt"
if [ -f "$TARGET_TS" ]; then
    SIZE=$(stat --format=%s "$TARGET_TS" 2>/dev/null || echo 0)
    if [ "$SIZE" -ge "$MIN_BYTES" ]; then
        echo "  ✓ TinyStories already present ($(du -h "$TARGET_TS" | cut -f1))."
    else
        echo "  ⚠ TinyStories exists but only ${SIZE} bytes — re-downloading."
        rm -f "$TARGET_TS"
    fi
fi

if [ ! -f "$TARGET_TS" ]; then
    echo "── Downloading TinyStories (~1.5 GB JSONL) ──"
    echo "  hf download roneneldan/TinyStories ..."
    echo ""

    hf download \
        roneneldan/TinyStories \
        TinyStories-train.txt \
        --repo-type dataset \
        --local-dir "$DATA_DIR"

    # Rename .txt → .txt so the streamer detects it as plain text
    if [ -f "$DATA_DIR/TinyStories-train.txt" ]; then
        mv "$DATA_DIR/TinyStories-train.txt" "$TARGET_TS"
    fi

    # Sanity check
    SIZE=$(stat --format=%s "$TARGET_TS" 2>/dev/null || echo 0)
    if [ "$SIZE" -lt "$MIN_BYTES" ]; then
        echo "✗ TinyStories download failed: only ${SIZE} bytes (expected > 10 MB)."
        echo "  Check: ls -lh $TARGET_TS"
        rm -f "$TARGET_TS"
        exit 1
    fi
    LINES=$(wc -l < "$TARGET_TS")
    echo "  ✓ TinyStories: $(du -h "$TARGET_TS" | cut -f1), $LINES JSONL lines."
fi

# ============================================================================
# 2. WikiText-103 — semantic co-occurrence (~500 MB raw)
# ============================================================================
TARGET_WT="$DATA_DIR/wiki.train.raw"
TARGET_TOKENS="$DATA_DIR/wiki.train.tokens"

if [ -f "$TARGET_TOKENS" ]; then
    SIZE=$(stat --format=%s "$TARGET_TOKENS" 2>/dev/null || echo 0)
    if [ "$SIZE" -ge "$MIN_BYTES" ]; then
        echo "  ✓ WikiText-103 already present ($(du -h "$TARGET_TOKENS" | cut -f1))."
    else
        echo "  ⚠ WikiText-103 exists but only ${SIZE} bytes — re-downloading."
        rm -f "$TARGET_TOKENS" "$TARGET_WT"
    fi
fi

if [ ! -f "$TARGET_TOKENS" ]; then
    echo ""
    echo "── Downloading WikiText-103 (training split only) ──"
    echo "  hf download Salesforce/wikitext --include '*train*' ..."
    echo ""

    hf download \
        Salesforce/wikitext \
        --include "*train*" \
        --repo-type dataset \
        --local-dir "$DATA_DIR" || true

    # Find wiki.train.raw wherever it landed and move it up
    FOUND=$(find "$DATA_DIR" -name "wiki.train.raw" -type f 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        echo "  Found wiki.train.raw at: $FOUND"
        mv "$FOUND" "$TARGET_WT"
        # Clean up WikiText subdirectories
        find "$DATA_DIR" -maxdepth 2 -type d -name "wikitech*" -exec rm -rf {} + 2>/dev/null || true
        find "$DATA_DIR" -maxdepth 2 -type d -name "wikitext*" -exec rm -rf {} + 2>/dev/null || true
    fi

    # Sanity check on raw download
    SIZE=$(stat --format=%s "$TARGET_WT" 2>/dev/null || echo 0)

    # ── Fallback: if WikiText still failed, use existing corpus ──
    if [ "$SIZE" -lt "$MIN_BYTES" ]; then
        echo "  ⚠ WikiText-103 LFS download incomplete (${SIZE} bytes)."
        echo "  Falling back to training_data/corpus.jsonl as secondary corpus..."
        rm -f "$TARGET_WT"
        # Extract all text from the existing 785 Wikipedia traces as fallback
        python3 << 'PYEOF'
import json, re

sentences = []
with open('training_data/corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ('system', 'thought', 'text', 'content'):
            text = rec.get(key)
            if text and isinstance(text, str) and len(text) > 50:
                # Split into sentences
                for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text):
                    s = s.strip()
                    tokens = [w for w in s.split() if any(c.isalpha() for c in w)]
                    if len(tokens) >= 5:
                        sentences.append(s)
                break

with open('training_data/wiki.train.tokens', 'w', encoding='utf-8') as f:
    for s in sentences:
        f.write(s + '\n')

print(f'  ✓ Fallback: {len(sentences)} sentences from corpus.jsonl.', file=sys.stderr)
PYEOF
        SIZE=$(stat --format=%s "$TARGET_TOKENS" 2>/dev/null || echo 0)
        if [ "$SIZE" -lt "$MIN_BYTES" ]; then
            echo "✗ Fallback also failed: only ${SIZE} bytes."
            rm -f "$TARGET_TOKENS"
            exit 1
        fi
        echo "  ✓ Fallback secondary corpus: $(du -h "$TARGET_TOKENS" | cut -f1)."
        # Skip the Wikitext conversion step
        skip_conversion=true
    else
        echo "  ✓ Raw download: $(du -h "$TARGET_WT" | cut -f1)."
        skip_conversion=false
    fi

    if [ "$skip_conversion" = false ]; then
        # Convert raw Wikitext → sentence-level plain text
        echo "  Converting WikiText-103 → sentence-level plain text (may take a minute)..."
        python3 << 'PYEOF'
import re, sys

with open('training_data/wiki.train.raw', 'r', encoding='utf-8', errors='replace') as f:
    raw = f.read()

raw = re.sub(r'(?m)^@.+$', '', raw)
raw = re.sub(r'(?m)^\s*=+.*?=+\s*$', '', raw)
raw = re.sub(r'<[^>]+>', '', raw)
raw = raw.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
raw = raw.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
raw = re.sub(r'https?://\S+', '', raw)
raw = re.sub(r'[ \t]+', ' ', raw)
raw = re.sub(r'\n{3,}', '\n\n', raw)

ABBREV = re.compile(r'\b(?:Dr|Mr|Mrs|Ms|St|Jr|Sr|vs|etc|approx|dept|est|govt)\.\s*$', re.I)

sentences = []
for para in raw.split('\n'):
    para = para.strip()
    if len(para) < 30:
        continue
    for part in re.split(r'(?<=[.!?])\s+(?=[A-Z"])', para):
        part = part.strip()
        if not part:
            continue
        if ABBREV.search(part):
            continue
        tokens = [w for w in part.split() if any(c.isalpha() for c in w)]
        if len(tokens) >= 5:
            sentences.append(part)

with open('training_data/wiki.train.tokens', 'w', encoding='utf-8') as f:
    for s in sentences:
        f.write(s + '\n')

print(f'  ✓ Converted: {len(sentences)} sentences.', file=sys.stderr)
PYEOF

        # Clean up raw
        rm -f "$TARGET_WT"

        # Final sanity
        LINES=$(wc -l < "$TARGET_TOKENS")
        SIZE=$(stat --format=%s "$TARGET_TOKENS" 2>/dev/null || echo 0)
        if [ "$LINES" -lt 100 ] || [ "$SIZE" -lt "$MIN_BYTES" ]; then
            echo "✗ WikiText-103 conversion produced only $LINES lines / ${SIZE} bytes."
            echo "  Something went wrong in the cleanup step."
            rm -f "$TARGET_TOKENS"
            exit 1
        fi
        echo "  ✓ WikiText-103: $(du -h "$TARGET_TOKENS" | cut -f1), $LINES sentences."
    fi
fi

# ============================================================================
# 3. Summary
# ============================================================================
echo ""
echo "── Dataset Summary ──"
for f in "$DATA_DIR"/TinyStories.txt "$DATA_DIR"/wiki.train.tokens; do
    if [ -f "$f" ]; then
        sz=$(du -h "$f" | cut -f1)
        ln=$(wc -l < "$f")
        printf "  %-25s %8s  (%d lines)\n" "$(basename "$f")" "$sz" "$ln"
    fi
done
echo ""
echo "✓ Datasets ready."
