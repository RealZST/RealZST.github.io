#!/bin/bash
# Build one note into the Jekyll collection.
#
#   bash scripts/build-note.sh roofline [/path/to/llm-infra-notes]
#
# Sources: _notes_src/<slug>/{post.zh.md,post.en.md,meta.yml} in this repository,
# and the figures in <llm-infra-notes>/<slug>/figures/ (the experiment repository).
# Writes _notes/<slug>.zh.md, _notes/<slug>.en.md and the figures the note uses to
# assets/notes/<slug>/.
# Nothing is committed.
set -euo pipefail
SITE=$(cd "$(dirname "$0")/.." && pwd)
SLUG=${1:?note slug}
EXPERIMENTS=${2:-$HOME/projects/llm-infra-notes}
SRC="$SITE/_notes_src/$SLUG"
[[ -f "$SRC/meta.yml" ]] || { echo "missing $SRC/meta.yml" >&2; exit 1; }
[[ -d "$EXPERIMENTS/$SLUG/figures" ]] || { echo "missing $EXPERIMENTS/$SLUG/figures" >&2; exit 1; }
mkdir -p "$SITE/_notes" "$SITE/assets/notes/$SLUG"
python3 - "$SRC" "$SITE" "$SLUG" "$EXPERIMENTS" <<'PY'
import pathlib, re, shutil, sys
import yaml
src, site, slug = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
figs = pathlib.Path(sys.argv[4]) / slug / "figures"
# copy only the figures the note uses
used = set()
for lang in ("zh", "en"):
    f = src / f"post.{lang}.md"
    if f.exists():
        used |= set(re.findall(r"\]\(figures/([^)]+)\)", f.read_text()))
for name in sorted(used):
    shutil.copy2(figs / name, site / "assets" / "notes" / slug / name)
meta = yaml.safe_load((src / "meta.yml").read_text())
for lang in ("zh", "en"):
    path = src / f"post.{lang}.md"
    if not path.exists():
        print("skip", path); continue
    body = path.read_text()
    body = re.sub(r"^# .*\n+", "", body, count=1)             # title lives in front matter
    # image + italic caption line -> <figure class="fig">; figures listed under
    # wide_figures in meta.yml (multi-panel ones) get fig-wide
    def figure(m):
        alt, name, cap = m.group(1), m.group(2), m.group(3)
        wide = " fig-wide" if name in meta.get("wide_figures", []) else ""
        cap = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", cap)  # the caption is raw HTML
        cap = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', cap)
        return (f'<figure class="fig{wide}"><img src="{{{{ \'/assets/notes/{slug}/{name}\' | relative_url }}}}" alt="{alt}" loading="lazy">'
                f'<figcaption>{cap}</figcaption></figure>\n')
    body = re.sub(r"!\[([^\]]*)\]\(figures/([^)]+)\)\n\n\*([^\n]+)\*\n", figure, body)
    m = meta[lang]
    fm = ["---", "layout: note", f"title: \"{m['title']}\"", f"date: {meta['date']}", f"lang: {lang}",
          f"slug: {slug}", f"summary: \"{m['summary']}\"",
          f"permalink: /notes/{slug}/" + ("" if lang == "en" else "zh/"), "---", ""]
    out = site / "_notes" / f"{slug}.{lang}.md"
    out.write_text("\n".join(fm) + body)
    print("wrote", out)
PY
echo "figures -> $SITE/assets/notes/$SLUG/"
