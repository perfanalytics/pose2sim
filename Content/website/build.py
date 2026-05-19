#!/usr/bin/env python3
"""Build Content/website/index.html from README.md.

Usage (from any directory):
    python Content/website/build.py
"""

import re
import sys
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
GITHUB_RAW  = "https://raw.githubusercontent.com/perfanalytics/pose2sim/main/"

try:
    import pymdownx  # noqa: F401
    MD_EXTENSIONS = ["tables", "pymdownx.superfences", "md_in_html", "sane_lists", "attr_list"]
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


# ---------------------------------------------------------------------------
# Preprocessing (applied to each section's markdown before conversion)
# ---------------------------------------------------------------------------

# Map admonition type → (CSS class, label)
ADMON_CSS  = {"NOTE": "info-box",    "TIP": "success-box",
              "WARNING": "warning-box", "IMPORTANT": "warning-box"}
ADMON_ICON = {"NOTE": "📝 Note", "TIP": "💡 Tip",
              "WARNING": "⚠️ Warning", "IMPORTANT": "❗ Important"}


def _replace_admonition(m: re.Match) -> str:
    atype   = m.group(1).upper()
    raw     = m.group(2)
    # Strip leading '> ' from each continuation line (handle both '> ' and '>')
    body    = re.sub(r"^>[ \t]?", "", raw, flags=re.MULTILINE).strip()
    css     = ADMON_CSS.get(atype, "info-box")
    label   = ADMON_ICON.get(atype, atype.capitalize())
    # Use markdown="1" so inner markdown is still processed by md_in_html
    return (
        f'\n<div class="{css}" markdown="1">\n\n'
        f'**{label}:**\n\n{body}\n\n'
        f"</div>\n"
    )


def preprocess(text: str) -> str:
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
        rf"\1{GITHUB_RAW}Content/", text
    )
    text = re.sub(
        r"!\[([^\]]*)\]\(Content/([^)]+)\)",
        rf"![\1]({GITHUB_RAW}Content/\2)", text
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
    # Capture content before the first heading (badges etc.)
    m = re.search(r"^# ", text, flags=re.MULTILINE)
    pre_h1 = text[:m.start()].strip() if m else ""
    text = text[m.start():] if m else text

    # Remove the "# Contents" section (auto-generated TOC)
    m_cont = re.search(r"^# Contents\s*$", text, flags=re.MULTILINE)
    if m_cont:
        m_next = re.search(r"^# ", text[m_cont.start() + 1:], flags=re.MULTILINE)
        if m_next:
            text = text[:m_cont.start()] + text[m_cont.start() + 1 + m_next.start():]

    heading_pat = re.compile(r"^(#{1,2}) +(.+)$", re.MULTILINE)
    matches = list(heading_pat.finditer(text))

    sections = []
    for i, m in enumerate(matches):
        level   = len(m.group(1))
        title   = m.group(2).strip()
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # Collect h3/h4 sub-headings inside this section (for sidebar TOC)
        children = []
        for sh in re.finditer(r"^(#{3,4}) +(.+)$", content, re.MULTILINE):
            children.append(
                {
                    "level": len(sh.group(1)),
                    "title": sh.group(2).strip(),
                    "slug":  slugify(sh.group(2).strip()),
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

def section_to_html(sec: dict) -> str:
    md = sec["content"]

    # Add id anchors to h3/h4 sub-headings so sidebar sub-items can jump to them
    def add_heading_id(m: re.Match) -> str:
        hashes = m.group(1)
        title  = m.group(2).strip()
        slug   = slugify(title)
        lvl    = len(hashes)
        t      = clean_title(title)
        return f'<h{lvl} id="{slug}">{t}</h{lvl}>'

    md = re.sub(r"^(#{3,4}) +(.+)$", add_heading_id, md, flags=re.MULTILINE)

    # Apply general preprocessing
    md = preprocess(md)

    # Convert to HTML
    body = markdown.markdown(md, extensions=MD_EXTENSIONS, output_format="html", tab_length=2)

    # Wrap <pre><code> in .code-block for dark styling
    body = re.sub(
        r"<pre><code([^>]*)>(.*?)</code></pre>",
        r'<div class="code-block"><pre><code\1>\2</code></pre></div>',
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
    <title>Pose2Sim Documentation</title>
    <link rel="stylesheet" href="style.css">
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
        <div id="google_translate_element"></div>
    </div>

    <!-- Sidebar -->
    <aside class="sidebar">
        <div class="logo">
            <h2>Documentation</h2>
            <div class="docs-switcher">
                <a href="#" class="docs-switch active" title="Pose2Sim documentation">Pose2Sim</a>
                <span class="docs-switch-sep">|</span>
                <a href="https://github.com/davidpagnon/Sports2D/blob/main/README.md"
                   class="docs-switch" target="_blank" rel="noopener" title="Sports2D documentation">Sports2D</a>
                <span class="docs-switch-sep">|</span>
                <a href="https://github.com/davidpagnon/Pose2Sim_Blender/blob/main/README.md"
                   class="docs-switch" target="_blank" rel="noopener" title="Pose2Sim Blender add-on documentation">Blender</a>
            </div>
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
                <p>Pose2Sim Documentation &bull;
                   <a href="https://github.com/perfanalytics/pose2sim" target="_blank">GitHub</a> &bull;
                   <a href="#how-to-cite">Cite</a>
                </p>
                <p>Open-source markerless motion capture &bull; BSD-3-Clause License</p>
                <p>Website Created by AYL & DP with ❤️ for the Markerless Community</p>
            </div>
        </footer>
    </main>

    <script src="script.js"></script>
    <script>document.addEventListener('DOMContentLoaded', () => hljs.highlightAll());</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Reading {README_PATH} …")
    text = README_PATH.read_text(encoding="utf-8")

    sections = split_sections(text)
    print(f"Found {len(sections)} raw sections.")

    # Mark which sections become navigable steps
    steps = [s for s in sections if _is_step(s)]

    # Mark the first step so the template can add class="step active"
    if steps:
        steps[0]["_first_step"] = True

    step_slugs = {s["slug"] for s in steps}

    print(f"Navigable steps ({len(steps)}): " + ", ".join(
        s["title"][:25] for s in steps
    ))

    toc_html      = build_toc_html(sections)
    sections_html = "\n".join(
        section_to_html(s) for s in sections if _is_step(s)
    )

    html = HTML_TEMPLATE.format(toc=toc_html, sections=sections_html)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Done. Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
