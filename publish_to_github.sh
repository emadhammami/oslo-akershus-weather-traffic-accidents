#!/usr/bin/env bash
set -euo pipefail

# Create a NEW public repo from this verified project directory. It deliberately
# refuses to overwrite or push over any existing remote repository.
OWNER="emadhammami"
NAME="oslo-akershus-weather-traffic-accidents"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
command -v gh >/dev/null || { echo "Install GitHub CLI (gh)." >&2; exit 1; }
command -v git >/dev/null || { echo "Install git." >&2; exit 1; }
command -v python3 >/dev/null || { echo "Install Python 3." >&2; exit 1; }

if [[ -n "${GH_TOKEN:-}" || -n "${GITHUB_TOKEN:-}" ]]; then
  echo "GH_TOKEN/GITHUB_TOKEN overrides gh auth account. Unset it before running." >&2
  exit 1
fi

gh auth switch -h github.com -u "$OWNER"
CURRENT="$(gh api user --jq '.login')"
[[ "$CURRENT" == "$OWNER" ]] || { echo "Wrong GitHub account: $CURRENT" >&2; exit 1; }

# Prevent accidental inclusion of credentials or outputs in tracked files.
python3 - <<'PYSEC'
from pathlib import Path
import json,re
repo=Path.cwd()
notebook=json.loads((repo/'notebooks/weather_traffic_analysis_colab.ipynb').read_text())
assert all(not c.get('outputs') for c in notebook['cells'] if c.get('cell_type')=='code')
assert all(c.get('execution_count') is None for c in notebook['cells'] if c.get('cell_type')=='code')
for p in repo.rglob('*'):
    if not p.is_file() or '.git' in p.parts:continue
    if p.suffix in {'.py','.md','.toml','.sh','.txt','.yml','.example','.ipynb'}:
        t=p.read_text(encoding='utf-8')
        if re.search(r'gh[opusr]_[a-zA-Z0-9]{20,}', t):
            raise SystemExit(f"Potential GitHub credential in {p}")
        for m in re.finditer(r'(?i)(?:FROST_CLIENT_SECRET|FROST_CLIENT_ID)\s*=\s*[\x22\x27]([^\x22\x27]+)[\x22\x27]', t):
            if not m.group(1).startswith(('your-', 'example-')):
                raise SystemExit(f"Potential embedded Frost credential in {p}")
print('Publish preflight: sanitized notebook, no tracked outputs/credentials found.')
PYSEC

if gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  echo "Repository already exists: https://github.com/$OWNER/$NAME ; refusing overwrite." >&2
  exit 1
fi
if [[ -d .git ]]; then
  echo "Existing .git directory found. Refusing implicit push." >&2
  exit 1
fi

git init -b main
git add README.md pyproject.toml .gitignore .env.example .github src scripts tests notebooks publish_to_github.sh
# Print exact included files for a final human check.
git status --short
read -r -p "Create PUBLIC repo $OWNER/$NAME with the files shown above? [yes/NO] " CONFIRM
[[ "$CONFIRM" == "yes" ]] || { echo 'Cancelled; nothing was pushed.'; exit 0; }

git commit -m 'Initial public research case study: sanitized weather and traffic analytics'
gh repo create "$OWNER/$NAME" --public --description 'USN 2026 team case study: monthly weather and personal-injury traffic accidents in Oslo and Akershus' --source . --remote origin --push
printf '
Published: https://github.com/%s/%s
' "$OWNER" "$NAME"
