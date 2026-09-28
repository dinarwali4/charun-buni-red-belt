#!/bin/zsh
cd "$1"; shift
while read u; do f=$(basename "$u"); [ -s "$f" ] && continue
  for i in 1 2 3; do curl -sfL -m 600 -H "Authorization: Bearer $(cat ~/.edl_token)" -o "$f" "$u" && break; sleep 3; done
  echo "$(date +%T) $f $(stat -f %z "$f" 2>/dev/null)"
done
