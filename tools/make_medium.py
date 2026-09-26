#!/usr/bin/env python3
"""Build docs/medium.html (a copy-paste-friendly version of ARTICLE.md for Medium) and render every markdown table
as an image, because Medium has no tables. Needs: python-markdown, google-chrome (headless), ImageMagick.
usage: python tools/make_medium.py   (from the repo root)"""
import os, re, subprocess, markdown

SITE = "https://sikamikanikobg.github.io/qwen38-flash-next-3x3090/m"
DEVTO = "https://dev.to/sikamikanikobg/i-ran-a-125b-model-on-three-rtx-3090s-at-80-tokenss-by-teaching-llamacpp-which-experts-matter-4ioi"
REPO = "https://github.com/SikamikanikoBG/qwen38-flash-next-3x3090"
CAPTIONS = [  # one per markdown table, in order
    ("table_benchmarks", "Qwen's own model-card numbers: Flash-Next vs Qwen3.8-27B."),
    ("table_profile", "Where one decoded word spent its time with cold experts in CPU RAM."),
    ("table_quality", "Quality vs the 8-bit reference (KL divergence: lower is closer to the original)."),
    ("table_context", "Speed vs prompt length for the two profiles (mean of two runs)."),
    ("table_realworld", "A real assistant turn vs the benchmark."),
    ("table_cache", "Returning to a cached prompt after another role ran: one slot vs four."),
]
TABLE_CSS = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body{font-family:Inter,sans-serif;background:#fff;margin:0;padding:28px 32px;width:1180px;color:#0f172a}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:19px;border:1px solid #e2e8f0;border-radius:14px;overflow:hidden}
th{background:#f1f5f9;text-align:left;font-weight:700;padding:14px 16px;border-bottom:1px solid #e2e8f0;color:#334155}
td{padding:13px 16px;border-bottom:1px solid #eef2f6} tr:last-child td{border-bottom:none} tr:nth-child(even) td{background:#fafbfc}
code{font-size:.9em;background:#f1f5f9;padding:1px 5px;border-radius:5px}"""


def render_table(md, name, outdir, tmpdir):
    html = markdown.markdown(md, extensions=["tables"])
    page = os.path.join(tmpdir, name + ".html")
    open(page, "w").write(f"<!doctype html><html><head><meta charset='utf-8'><style>{TABLE_CSS}</style></head><body>{html}</body></html>")
    png = os.path.join(tmpdir, name + ".png")
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=1244,900", "--virtual-time-budget=4000", f"--screenshot={png}", "file://" + page],
                   check=True, stderr=subprocess.DEVNULL)
    subprocess.run(["magick", png, "-trim", "-bordercolor", "white", "-border", "40", "-quality", "88",
                    os.path.join(outdir, name + ".jpg")], check=True)


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    outdir = os.path.join(root, "docs", "m"); tmpdir = os.path.join(root, "docs", ".tmp")
    os.makedirs(outdir, exist_ok=True); os.makedirs(tmpdir, exist_ok=True)
    s = open(os.path.join(root, "ARTICLE.md")).read()
    fm, body = re.match(r"^---\n(.*?)\n---\n(.*)$", s, re.S).groups()
    meta = dict(l.split(": ", 1) for l in fm.splitlines())
    title, desc = meta["title"].strip('"'), meta["description"].strip('"')
    lines, out, i, k = body.splitlines(), [], 0, 0
    while i < len(lines):
        if lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            name, cap = CAPTIONS[k]; k += 1
            render_table("\n".join(lines[i:j]), name, outdir, tmpdir)
            out.append(f"\n@@FIG {name}.jpg|{cap}@@\n"); i = j; continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)\s*$", lines[i])
        if m:
            alt, url = m.groups()
            out.append(f"\n@@FIG {url.rsplit('/', 1)[1].replace('.png', '.jpg')}|{alt}@@\n"); i += 1; continue
        out.append(lines[i]); i += 1
    assert k == len(CAPTIONS), f"{k} tables in the article, {len(CAPTIONS)} captions"
    html = markdown.markdown("\n".join(out), extensions=["fenced_code"])
    html = re.sub(r"<p>@@FIG (.*?)\|(.*?)@@</p>",
                  lambda m: f'<figure><img src="{SITE}/{m.group(1)}" alt="{m.group(2)}"><figcaption>{m.group(2)}</figcaption></figure>', html)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{max-width:760px;margin:40px auto;padding:0 20px;font-family:Georgia,serif;line-height:1.6;color:#1a1a1a}}
img{{max-width:100%}} figure{{margin:28px 0}} figcaption{{font-size:14px;color:#666;text-align:center}}
pre{{background:#f5f5f5;padding:12px;overflow-x:auto}} .tip{{font-family:sans-serif;background:#fff7e6;border:1px solid #f5d38a;padding:12px 16px;border-radius:8px;font-size:15px}}</style></head>
<body>
<div class="tip">For Medium: click just before the title, scroll to the end, <b>Shift+click</b> after the last line, <b>Ctrl+C</b>, then paste into a new Medium story (medium.com/new-story).</div>
<article>
<h1>{title}</h1>
<h2>{desc}</h2>
<figure><img src="{SITE}/cover-qwen38-flash-next.jpg" alt="Qwen3.8-Flash-Next on three RTX 3090s at 80 tok/s"></figure>
{html}
<p><em>Originally published on <a href="{DEVTO}">DEV</a>. Code and data: <a href="{REPO}">{REPO.replace('https://', '')}</a>.</em></p>
</article></body></html>"""
    for name in ("medium.html", "index.html"):  # index.html so the bare site URL works too
        open(os.path.join(root, "docs", name), "w").write(page)
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)
    print(f"docs/medium.html: {k} tables as images, {page.count('<figure>')} figures")


if __name__ == "__main__":
    main()
