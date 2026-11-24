#!/usr/bin/env python3
"""Build the documentation website from README.md.

Outputs
  Content/website/index.html              README.md of the latest commit @ main
  Content/website/v/<tag>/index.html      README.md as it was at each GitHub release

Usage (from any directory):
    python Content/website/build.py                  # bleeding edge + all releases
    python Content/website/build.py --no-versions    # bleeding edge only (fast, offline)
    python Content/website/build.py --limit 10       # last 10 releases only

Building the release pages needs the full git history and tags
(`actions/checkout` with `fetch-depth: 0`) and, ideally, a GITHUB_TOKEN
environment variable so the GitHub API does not rate-limit the release listing.
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    import markdown
except ImportError:
    print("Please install: pip install markdown", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
README_PATH = REPO_ROOT / "README.md"
OUTPUT_PATH = SCRIPT_DIR / "index.html"
REPO_SLUG   = "perfanalytics/pose2sim"
DOC_NAME    = "Pose2Sim"                 # shown in page titles
VERSIONS_DIR = SCRIPT_DIR / "v"          # one sub-folder per release
VERSIONS_URL = "v"


def raw_base(ref: str) -> str:
    """Base URL for raw files (images...) at a given branch / tag / commit."""
    return f"https://raw.githubusercontent.com/{REPO_SLUG}/{ref}/"

try:
    import pymdownx  # noqa: F401
    MD_EXTENSIONS = ["tables", "pymdownx.superfences", "pymdownx.tasklist", "md_in_html", "sane_lists", "attr_list"]
except ImportError:
    MD_EXTENSIONS = ["tables", "fenced_code", "md_in_html", "sane_lists", "attr_list"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    """GitHub-compatible heading slug."""
    text = re.sub(r"<[^>]+>", "", text)           # strip HTML tags
    text = re.sub(r"[^\w\s-]", "", text.lower())  # keep word chars, spaces, -
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def clean_title(text: str) -> str:
    """Strip markdown formatting from a heading for display."""
    text = re.sub(r"\*+|_+|`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) → label
    return text.strip()

FENCE_RE = re.compile(
    r"^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

def mask_code_fences(text: str) -> str:
    """Blank out fenced code blocks (same length, newlines kept) so that
    regexes searching for headings don't match '#' comments inside code."""
    return FENCE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


# ---------------------------------------------------------------------------
# Preprocessing (applied to each section's markdown before conversion)
# ---------------------------------------------------------------------------

# Map admonition type → (CSS class, label)
ADMON_CSS  = {"NOTE": "info-box",    "TIP": "success-box",
              "WARNING": "warning-box", "IMPORTANT": "important-box"}
ADMON_ICON = {"NOTE": "📝 Note", "TIP": "💡 Tip",
              "WARNING": "⚠️ Warning", "IMPORTANT": "❗ Important"}


def add_heading_ids(md: str) -> str:
    """Turn ### / #### headings into <hN id=...> with a clickable # anchor."""
    masked = mask_code_fences(md)
    out, last = [], 0
    for m in re.finditer(r"^(#{3,4}) +(.+)$", masked, flags=re.MULTILINE):
        lvl   = len(m.group(1))
        title = md[m.start(2):m.end(2)].strip()
        slug  = slugify(title)
        t     = clean_title(title)
        out.append(md[last:m.start()])
        out.append(f'<h{lvl} id="{slug}">{t}'
                   f'<a href="#{slug}" class="heading-anchor" aria-label="Link to section">#</a>'
                   f'</h{lvl}>')
        last = m.end()
    out.append(md[last:])
    return "".join(out)


def _replace_admonition(m: re.Match) -> str:
    atype = m.group(1).upper()
    raw   = m.group(2)
    body  = re.sub(r"^>[ \t]?", "", raw, flags=re.MULTILINE).strip()
    css   = ADMON_CSS.get(atype, "info-box")
    label = ADMON_ICON.get(atype, atype.capitalize())

    body = add_heading_ids(body)          # <-- new: give headings ids/anchors first
    body_html = markdown.markdown(body, extensions=MD_EXTENSIONS, tab_length=2)

    return (
        f'\n<div class="{css}">\n'
        f'<h4>{label}:</h4>\n'
        f'{body_html}\n'
        f'</div>\n'
    )


def preprocess(text: str, raw: str) -> str:
    """Transform README markdown before passing to the markdown library."""

    # 0. Normalize Windows line endings
    text = text.replace('\r\n', '\n')

    # 0b. Backslash line-breaks (GitHub extension: \<newline>) → <br>
    text = re.sub(r'\\\n', '<br>\n', text)

    # 0c. Normalize code block language tags for highlight.js
    lang_remaps = {'cmd': 'bash', 'powershell': 'bash', 'zsh': 'bash', 'sh': 'bash', 'shell': 'bash'}
    def remap_lang(m):
        lang = m.group(1).strip().lower()
        lang = lang_remaps.get(lang, lang)
        return f'```{lang}'
    text = re.sub(r'```(\w+)', remap_lang, text)
    
    # 1. Fix relative image paths → raw GitHub URLs
    text = re.sub(
        r'(<img\b[^>]*\bsrc=")Content/',
        rf"\1{raw}Content/", text
    )
    # 1b. Remove fixed width/height attrs from <img> so CSS can make them responsive
    text = re.sub(r'(<img\b[^>]*?)\s+width="\d+"', r'\1', text)
    text = re.sub(r'(<img\b[^>]*?)\s+height="\d+"', r'\1', text)
    # 1c. Mark explicit <img> HTML tags as block-level content images (not inline badge images)
    text = re.sub(r'(<img\b)(?![^>]*\bclass=)', r'\1 class="img-block"', text)
    text = re.sub(
        r"!\[([^\]]*)\]\(Content/([^)]+)\)",
        rf"![\1]({raw}Content/\2)", text
    )

    # 2. Bare video/GitHub asset URLs on their own line → <video>
    text = re.sub(
        r"^(https://\S+\.(?:mp4|webm|mov|MP4))\s*$",
        r'<video controls style="max-width:100%;border-radius:8px;margin:20px 0">'
        r'<source src="\1" type="video/mp4"></video>',
        text, flags=re.MULTILINE,
    )
    # GitHub user-attachments video links
    text = re.sub(
        r"^(https://github\.com/user-attachments/assets/[A-Za-z0-9_-]+)\s*$",
        r'<video controls style="max-width:100%;border-radius:8px;margin:20px 0">'
        r'<source src="\1" type="video/mp4"></video>',
        text, flags=re.MULTILINE,
    )

    # 3. GitHub admonitions  →  HTML divs (multi-line)
    # Handle blank blockquote lines that may be '>' with no trailing space
    text = re.sub(
        r"> \[!(NOTE|TIP|WARNING|IMPORTANT)\]\n((?:>[ \t]?[^\n]*\n?)+)",
        _replace_admonition, text,
    )

    # 4. <details> blocks → h3 heading + expanded content (remove CLICK TO SHOW)
    def _replace_details(m: re.Match) -> str:
        title   = m.group(1).strip()
        content = m.group(2).strip()
        # Remove <pre>/<pre> wrappers so inner content renders as markdown/HTML
        content = re.sub(r'\s*</?pre>\s*', '\n\n', content).strip()
        return f'\n### {title}\n\n{content}\n\n'

    text = re.sub(
        r'<details[^>]*>\s*<summary[^>]*><b>([^<]+)</b>[^<]*</summary>(.*?)</details>',
        _replace_details, text, flags=re.DOTALL
    )

    return text


# ---------------------------------------------------------------------------
# Splitting README into sections
# ---------------------------------------------------------------------------

def split_sections(text: str) -> list[dict]:
    """
    Split README.md at h1/h2 heading boundaries.
    Returns list of dicts:
      level, title, slug, content (raw markdown), children (list of h3/h4 dicts)
    """
    # Strip the "Visit the website!" preamble
    sep = re.search(r"<br>\s*\n\n---\n\n<br>", text)
    if sep:
        text = text[sep.end():].lstrip("\n")

    masked = mask_code_fences(text)

    # Capture content before the first heading (badges etc.)
    m = re.search(r"^# ", masked, flags=re.MULTILINE)
    pre_h1 = text[:m.start()].strip() if m else ""
    if m:
        text, masked = text[m.start():], masked[m.start():]

    # Remove the "# Contents" section (auto-generated TOC)
    m_cont = re.search(r"^# Contents\s*$", masked, flags=re.MULTILINE)
    if m_cont:
        m_next = re.search(r"^# ", masked[m_cont.start() + 1:], flags=re.MULTILINE)
        if m_next:
            cut    = m_cont.start() + 1 + m_next.start()
            text   = text[:m_cont.start()]   + text[cut:]
            masked = masked[:m_cont.start()] + masked[cut:]

    heading_pat = re.compile(r"^(#{1,2}) +(.+)$", re.MULTILINE)
    matches = list(heading_pat.finditer(masked))

    sections = []
    for i, m in enumerate(matches):
        level   = len(m.group(1))
        title   = text[m.start(2):m.end(2)].strip()
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # Collect h3/h4 sub-headings (for sidebar TOC), ignoring code fences
        children = []
        cmasked = mask_code_fences(content)
        for sh in re.finditer(r"^(#{3,4}) +(.+)$", cmasked, re.MULTILINE):
            ctitle = content[sh.start(2):sh.end(2)].strip()
            children.append(
                {
                    "level": len(sh.group(1)),
                    "title": ctitle,
                    "slug":  slugify(ctitle),
                }
            )

        sections.append(
            {
                "level":    level,
                "title":    title,
                "slug":     slugify(title),
                "content":  content,
                "children": children,
            }
        )

    # Prepend pre-heading content (badges, etc.) to the first section
    if pre_h1 and sections:
        sections[0]["content"] = pre_h1 + "\n\n" + sections[0]["content"]

    return sections


# ---------------------------------------------------------------------------
# Sidebar TOC
# ---------------------------------------------------------------------------

def build_toc_html(sections: list[dict]) -> str:
    parts = []
    for i, sec in enumerate(sections):
        t = clean_title(sec["title"])
        if sec["level"] == 1:
            if _is_step(sec):
                target_slug = sec["slug"]
            else:
                # Link to the first h2 child so clicking always navigates somewhere
                nxt = next((s for s in sections[i + 1:] if s["level"] == 2), None)
                target_slug = nxt["slug"] if nxt else None
            if target_slug:
                parts.append(
                    f'<a href="#{target_slug}" class="nav-item nav-h1" '
                    f'data-section="{target_slug}">{t}</a>'
                )
        else:  # h2 → indented nav item
            parts.append(
                f'<a href="#{sec["slug"]}" class="nav-item nav-h2" '
                f'data-section="{sec["slug"]}">{t}</a>'
            )
    return "\n".join(parts)


def _is_step(sec: dict) -> bool:
    """h1 sections with substantial content become navigable steps."""
    return sec["level"] == 2 or len(sec["content"].strip()) > 80


# ---------------------------------------------------------------------------
# Markdown → HTML conversion for one section
# ---------------------------------------------------------------------------

def section_to_html(sec: dict, raw: str) -> str:
    md = sec["content"]

    # Apply general preprocessing
    md = preprocess(md, raw)

    # Add id anchors to h3/h4 sub-headings so sidebar sub-items can jump to them
    md = add_heading_ids(md)
    
    # Convert to HTML
    body = markdown.markdown(md, extensions=MD_EXTENSIONS, output_format="html", tab_length=2)

    # Wrap <pre><code> in .code-block for dark styling
    body = re.sub(
        r'(?:<div class="highlight">)?(<pre(?:\s[^>]*)?>.*?</pre>)(?:</div>)?',
        r'<div class="code-block">\1</div>',
        body, flags=re.DOTALL,
    )

    # Wrap bare <table> in .params-table for styling
    body = body.replace("<table>", '<div class="params-table"><table>').replace(
        "</table>", "</table></div>"
    )

    title = clean_title(sec["title"])
    active = ' active' if sec.get("_first_step") else ''

    return (
        f'<section class="step{active}" id="{sec["slug"]}">\n'
        f'  <div class="step-header"><h1>{title}</h1></div>\n'
        f'  <div class="step-content">\n{body}\n  </div>\n'
        f'</section>\n'
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
{robots_meta}
    <link rel="icon" type="image/x-icon" href="{site_root}Pose2Sim_favicon.ico"> 
    <link rel="stylesheet" href="{style_href}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark-dimmed.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- Google Translate -->
    <script type="text/javascript">
        function googleTranslateElementInit() {{
            new google.translate.TranslateElement({{
                pageLanguage: 'en',
                includedLanguages: 'en,fr,es,de,it,pt,zh-CN,zh-TW,ja,ko,ar,ru,hi,tr,nl,pl,sv,da,fi,no,cs,el,he,id,th,vi,uk,ro,hu,sk,bg,hr,sr,sl,lt,lv,et',
                layout: google.translate.TranslateElement.InlineLayout.SIMPLE,
                autoDisplay: false
            }}, 'google_translate_element');
        }}
    </script>
    <script type="text/javascript" src="https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
</head>
<body>

    <!-- Google Translate Widget -->
    <div class="translate-widget">
        <button class="theme-toggle-btn" id="themeToggleBtn" onclick="toggleTheme()">
            <span class="theme-icon" id="themeIcon">☀️</span>
            <span id="themeLabel">Light mode</span>
        </button>
        <div id="google_translate_element"></div>
    </div>

    <!-- Hamburger button (mobile only) -->
    <button class="hamburger" id="hamburgerBtn" onclick="toggleSidebar()" aria-label="Open navigation">
        <span></span><span></span><span></span>
    </button>
    <!-- Sidebar overlay (mobile) -->
    <div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>

    <!-- Sidebar -->
    <aside class="sidebar" id="sidebar">
        <div class="logo">
            <h2>Documentation</h2>
            <div class="docs-switcher">
                <a href="https://perfanalytics.github.io/pose2sim/#pose2sim" 
                   class="docs-switch active" title="Pose2Sim documentation">Pose2Sim</a>
                <span class="docs-switch-sep">|</span>
                <a href="https://davidpagnon.github.io/Sports2D/#sports2d"
                   class="docs-switch" title="Sports2D documentation">Sports2D</a>
                <span class="docs-switch-sep">|</span>
                <a href="https://davidpagnon.github.io/Pose2Sim_Blender/#pose2sim-blender"
                   class="docs-switch" title="Pose2Sim Blender add-on documentation">Blender add-on</a>
            </div>
{version_switcher}
        </div>
        <nav class="nav-menu">
{toc}
        </nav>
        <button class="view-all-btn" onclick="toggleViewAll()">
            <span>View All</span>
        </button>
    </aside>

    <!-- Main Content -->
    <main class="content">
{version_banner}
        <div class="step-container">
{sections}
        </div>

        <!-- Navigation Buttons -->
        <div class="nav-buttons">
            <button class="btn btn-secondary" onclick="previousSection()" id="prevBtn" style="display:none;">
                <span>← Previous</span>
            </button>
            <button class="btn btn-primary" onclick="nextSection()" id="nextBtn">
                <span>Next →</span>
            </button>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <div class="footer-content">
                <p><a href="https://github.com/perfanalytics/pose2sim" target="_blank">Pose2Sim GitHub</a> (⭐Star it⭐ or ❤️<a href="https://github.com/sponsors/davidpagnon" target="_blank">Sponsor it!</a>❤️) &bull;
                   <a href="#how-to-cite">Cite Pose2Sim</a>
                </p>
                <p>Open-source markerless motion capture &bull; BSD-3-Clause License</p>
                <p>Website Created by AYL & <a href="https://david-pagnon.com/fr/en-deux-mots/" target="_blank">DP</a> with ❤️ for the Markerless Community</p>
            </div>
        </footer>
    </main>

    <script src="{script_src}"></script>
    <script>document.addEventListener('DOMContentLoaded', () => hljs.highlightAll());</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Versions (GitHub releases)
# ---------------------------------------------------------------------------

def git(*args: str) -> str | None:
    """Run a git command in the repo; return stdout, or None on failure."""
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                       text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def fetch_releases(include_prereleases: bool = False) -> list[dict]:
    """List published GitHub releases (newest first) through the GitHub API."""
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "pose2sim-docs-build"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    releases, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{REPO_SLUG}/releases?per_page=100&page={page}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.load(resp)
        if not batch:
            break
        releases += batch
        page += 1

    out = []
    for r in releases:
        if r.get("draft") or (r.get("prerelease") and not include_prereleases):
            continue
        out.append({
            "tag":  r["tag_name"],
            "date": r["published_at"],            # ISO 8601, UTC
            "slug": re.sub(r"[^\w.\-]", "_", r["tag_name"]),
        })
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


def readme_at_release(rel: dict) -> tuple[str, str] | None:
    """Return (README text, git ref used for raw image URLs) for a release.

    Uses the README.md stored in the release's tag. If the tag is not
    available locally, falls back to the last commit made before the release
    date.
    """
    ref = f"refs/tags/{rel['tag']}"
    text = git("show", f"{ref}:README.md")
    if text is not None:
        return text, rel["tag"]

    sha = (git("rev-list", "-1", f"--before={rel['date']}", "HEAD") or "").strip()
    if sha:
        text = git("show", f"{sha}:README.md")
        if text is not None:
            print(f"  note: tag {rel['tag']} not found, using last commit before "
                  f"{rel['date']} ({sha[:7]})")
            return text, sha
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_readme(text: str, raw: str) -> tuple[str, str]:
    """README markdown -> (sidebar TOC html, sections html)."""
    sections = split_sections(text)
    steps = [s for s in sections if _is_step(s)]
    if not steps:
        raise ValueError("no navigable sections found in README")
    steps[0]["_first_step"] = True
    return build_toc_html(sections), "\n".join(section_to_html(s, raw) for s in steps)


def version_href(slug: str, site_root: str) -> str:
    """URL of a version's page, relative to the page being generated.
    Always ends in index.html: directory URLs only work on a web server,
    not when the site is opened from disk (file://)."""
    if slug == "latest":
        return f"{site_root}index.html"
    return f"{site_root}{VERSIONS_URL}/{slug}/index.html"


def version_switcher_html(versions: list[dict], current: str, site_root: str) -> str:
    """<select> listing every version; option values are URLs relative to the
    page being generated."""
    if len(versions) < 2:
        return ""
    opts = []
    for v in versions:
        sel = " selected" if v["slug"] == current else ""
        opts.append(f'<option value="{html.escape(version_href(v["slug"], site_root))}"{sel}>'
                    f'{html.escape(v["label"])}</option>')
    return (
        '<div class="version-switcher">\n'
        '  <select id="versionSelect" onchange="switchVersion(this)" aria-label="Documentation version">\n    '
        + "\n    ".join(opts) +
        '\n  </select>\n</div>'
    )


def write_page(out_path: Path, *, toc: str, sections: str, site_root: str,
               versions: list[dict], current: dict) -> None:
    is_latest = current["slug"] == "latest"
    asset_version = int(max(
        (SCRIPT_DIR / "script.js").stat().st_mtime,
        (SCRIPT_DIR / "style.css").stat().st_mtime,
    ))

    if is_latest:
        title, robots, banner = f"{DOC_NAME} Documentation", "", ""
    else:
        title  = f"{DOC_NAME} Documentation ({current['tag']})"
        robots = '    <meta name="robots" content="noindex">'
        banner = (
            '<div class="version-banner">'
            f'You are reading the documentation for release <strong>{html.escape(current["tag"])}</strong> '
            f'({current["date"][:10]}). '
            f'<a href="{site_root}index.html" onclick="this.href=this.getAttribute(\'href\').split(\'#\')[0]+location.hash">Go to the latest version</a>.'
            '</div>'
        )

    page = HTML_TEMPLATE.format(
        title=title,
        robots_meta=robots,
        site_root=site_root,
        toc=toc,
        sections=sections,
        version_switcher=version_switcher_html(versions, current["slug"], site_root),
        version_banner=banner,
        style_href=f"{site_root}style.css?v={asset_version}",
        script_src=f"{site_root}script.js?v={asset_version}",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_versions_index() -> None:
    """v/index.html: someone browsing to the parent folder lands on the latest docs."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (VERSIONS_DIR / "index.html").write_text(
        '<!DOCTYPE html>\n<meta charset="utf-8">\n'
        f'<title>{DOC_NAME} documentation</title>\n'
        '<meta name="robots" content="noindex">\n'
        '<meta http-equiv="refresh" content="0; url=../index.html">\n'
        "<script>location.replace('../index.html' + location.hash);</script>\n"
        f'<p><a href="../index.html">Go to the {DOC_NAME} documentation</a></p>\n',
        encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-versions", action="store_true",
                    help="only build the bleeding-edge page (no GitHub API / git history needed)")
    ap.add_argument("--include-prereleases", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="only keep the N most recent releases (0 = all)")
    args = ap.parse_args()

    # 1. Bleeding edge: README of the checked-out commit ---------------------
    print(f"Reading {README_PATH} …")
    latest_text = README_PATH.read_text(encoding="utf-8")
    latest = {"slug": "latest", "tag": "latest", "date": "",
              "label": "main"}
    latest_toc, latest_sections = render_readme(latest_text, raw_base("main"))

    # 2. Releases ---------------------------------------------------------------
    built: list[tuple[dict, str, str]] = []     # (version, toc, sections)
    if not args.no_versions:
        try:
            releases = fetch_releases(args.include_prereleases)
        except Exception as exc:                # network / rate limit
            print(f"WARNING: could not list releases ({exc}); building latest only.",
                  file=sys.stderr)
            releases = []
        if args.limit:
            releases = releases[:args.limit]

        for rel in releases:
            got = readme_at_release(rel)
            if got is None:
                print(f"  skip {rel['tag']}: no README.md found", file=sys.stderr)
                continue
            text, ref = got
            try:
                toc, secs = render_readme(text, raw_base(ref))
            except Exception as exc:            # old README with an odd structure
                print(f"  skip {rel['tag']}: {exc}", file=sys.stderr)
                continue
            rel["label"] = rel["tag"]
            built.append((rel, toc, secs))
            print(f"  built {rel['tag']}")

    versions = [latest] + [b[0] for b in built]

    # 3. Write everything -------------------------------------------------------
    if VERSIONS_DIR.exists():
        shutil.rmtree(VERSIONS_DIR)             # drop pages of removed releases
    write_page(OUTPUT_PATH, toc=latest_toc, sections=latest_sections,
               site_root="", versions=versions, current=latest)
    if built:
        write_versions_index()
    for rel, toc, secs in built:
        write_page(VERSIONS_DIR / rel["slug"] / "index.html", toc=toc, sections=secs,
                   site_root="../../", versions=versions, current=rel)

    print(f"Done. Latest -> {OUTPUT_PATH}; {len(built)} release page(s) -> {VERSIONS_DIR}")


if __name__ == "__main__":
    main()
