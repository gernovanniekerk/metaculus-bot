#!/usr/bin/env bash
# Commit the spend ledger (and any saved comments awaiting retry) back to the
# repo so the next run knows this month's spend. Runs after every bot run, even
# a failed one. Fails loudly if the push does not land, because a lost ledger
# update would let the next run under-count spend.
set -euo pipefail

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add state/
if git diff --cached --quiet; then
  echo "Spend ledger unchanged."
  exit 0
fi
git commit -q -m "Update spend ledger [skip ci]"
for attempt in 1 2 3; do
  if git pull --rebase --quiet && git push --quiet; then
    echo "Spend ledger saved."
    exit 0
  fi
  echo "Push attempt ${attempt} failed; retrying."
  sleep $((attempt * 5))
done
echo "::error::Could not push the spend ledger. Check state/spend_ledger.json before the next run."
exit 1
