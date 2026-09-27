#!/usr/bin/env bash
set -e
mkdir -p data/attention
for pfx in $(ls data/attention_parts/*.part-* | sed 's/\.part-[^.]*$//' | sort -u); do
  out="data/attention/$(basename "$pfx" | sed 's/\.json\.gz$//').json"
  echo "rebuild $out"
  cat "$pfx".part-* | gzip -d > "$out"
done
ls -lh data/attention
