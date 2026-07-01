'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let sections    = [];   // Array of section IDs in DOM order
let currentIdx  = 0;    // Index of currently visible section
let viewAllMode = false;


function copyTextToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text);
    }

    return new Promise((resolve, reject) => {
        try {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.setAttribute('readonly', '');
            textarea.style.position = 'fixed';
            textarea.style.top = '-9999px';
            textarea.style.left = '-9999px';
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();

            const ok = document.execCommand('copy');
            document.body.removeChild(textarea);
            if (ok) resolve();
            else reject(new Error('Copy command was rejected.'));
        } catch (err) {
            reject(err);
        }
    });
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    sections = Array.from(document.querySelectorAll('.step'))
                    .map(el => el.id)
                    .filter(Boolean);

    if (sections.length === 0) return;

    const hash = location.hash.slice(1);
    const targetSection = resolveSection(hash);
    showSection(targetSection, hash === targetSection || !hash, hash || targetSection);
    if (hash && hash !== targetSection) {
        const el = document.getElementById(hash);
        if (el) setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    }
    updateNavButtons();
    updateActiveNav(targetSection);

    // Inject copy buttons into all code blocks
    document.querySelectorAll('.code-block').forEach(block => {
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.setAttribute('aria-label', 'Copy code');
        btn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span>Copy</span>`;
        btn.addEventListener('click', () => {
            const code = block.querySelector('code');
            const text = code ? code.innerText : block.innerText;
            copyTextToClipboard(text).then(() => {
                btn.classList.add('copied');
                btn.querySelector('svg').innerHTML = `<polyline points="20 6 9 17 4 12"></polyline>`;
                btn.querySelector('span').textContent = 'Copied!';
                setTimeout(() => {
                    btn.classList.remove('copied');
                    btn.querySelector('svg').innerHTML = `<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>`;
                    btn.querySelector('span').textContent = 'Copy';
                }, 2000);
            }).catch(() => {
                btn.querySelector('span').textContent = 'Copy failed';
                setTimeout(() => {
                    btn.querySelector('span').textContent = 'Copy';
                }, 1500);
            });
        });
        block.appendChild(btn);
    });

    // Intercept in-content anchor link clicks so hidden sections become visible
    document.addEventListener('click', e => {
        const link = e.target.closest('a[href^="#"]');
        if (!link) return;
        const hash = link.getAttribute('href').slice(1);
        if (!hash) return;

        // Is it a direct section ID?
        if (sections.includes(hash)) {
            e.preventDefault();
            goToSection(hash, null);
            return;
        }

        // Is the target ID inside a known section?
        const targetEl = document.getElementById(hash);
        if (targetEl) {
            const parentSection = targetEl.closest('.step');
            if (parentSection && parentSection.id) {
                e.preventDefault();
                goToSection(parentSection.id, hash);
                return;
            }
        }

        // Fall back to nav-item data-anchor lookup
        const navItem = document.querySelector(`.nav-item[data-anchor="${hash}"]`);
        if (navItem && navItem.dataset.section) {
            e.preventDefault();
            goToSection(navItem.dataset.section, hash);
        }
    });
});

// Also handle direct URL hash navigation (back/forward, typed URLs)
window.addEventListener('hashchange', () => {
    const hash = location.hash.slice(1);
    if (!hash) return;
    const targetSection = resolveSection(hash);
    showSection(targetSection, hash === targetSection, hash);
    if (hash !== targetSection) {
        const el = document.getElementById(hash);
        if (el) setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    }
    updateNavButtons();
    updateActiveNav(targetSection);
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function resolveSection(hash) {
    if (!hash) return sections[0];
    if (sections.includes(hash)) return hash;
    const navItem = document.querySelector(`.nav-item[data-anchor="${hash}"]`);
    if (navItem && navItem.dataset.section) return navItem.dataset.section;
    // Sub-anchor inside a section: find its parent .step
    const el = document.getElementById(hash);
    if (el) {
        const parent = el.closest('.step');
        if (parent && parent.id) return parent.id;
    }
    return sections[0];
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function goToSection(sectionId, anchor) {
    if (viewAllMode) toggleViewAll();
    showSection(sectionId, !anchor, anchor || sectionId);
    updateNavButtons();
    updateActiveNav(sectionId);
    if (anchor) {
        const el = document.getElementById(anchor);
        if (el) setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
    }
}

function showSection(sectionId, scrollTop = true, urlHash = null) {
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    const el = document.getElementById(sectionId);
    if (el) {
        el.classList.add('active');
        currentIdx = sections.indexOf(sectionId);
        history.replaceState(null, '', `#${urlHash || sectionId}`);
    }
    if (scrollTop) window.scrollTo({ top: 0, behavior: 'smooth' });
}

function nextSection() {
    if (viewAllMode) { toggleViewAll(); return; }
    if (currentIdx < sections.length - 1) goToSection(sections[currentIdx + 1], null);
}

function previousSection() {
    if (viewAllMode) { toggleViewAll(); return; }
    if (currentIdx > 0) goToSection(sections[currentIdx - 1], null);
}

// ---------------------------------------------------------------------------
// UI state updates
// ---------------------------------------------------------------------------
function updateNavButtons() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    if (!prevBtn || !nextBtn) return;

    if (viewAllMode) {
        prevBtn.style.display = 'none';
        nextBtn.querySelector('span').textContent = 'Exit View All';
        return;
    }

    prevBtn.style.display = currentIdx === 0 ? 'none' : 'inline-flex';
    nextBtn.querySelector('span').textContent = 'Next →';
    nextBtn.disabled = currentIdx === sections.length - 1;
}

function updateActiveNav(activeSectionId) {
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    document.querySelectorAll(`.nav-item[data-section="${activeSectionId}"]`)
            .forEach(item => item.classList.add('active'));
}

// ---------------------------------------------------------------------------
// View-all toggle
// ---------------------------------------------------------------------------
function toggleViewAll() {
    viewAllMode = !viewAllMode;
    const steps   = document.querySelectorAll('.step');
    const btnSpan = document.querySelector('.view-all-btn span');

    if (viewAllMode) {
        steps.forEach(s => s.classList.add('view-all-mode', 'active'));
        if (btnSpan) btnSpan.textContent = 'Back to Step View';
    } else {
        steps.forEach(s => s.classList.remove('view-all-mode', 'active'));
        showSection(sections[currentIdx], true);
        if (btnSpan) btnSpan.textContent = 'View All';
    }

    updateNavButtons();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---------------------------------------------------------------------------
// Dark / Light theme
// ---------------------------------------------------------------------------
function applyTheme(dark) {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : '');
    const icon  = document.getElementById('themeIcon');
    const label = document.getElementById('themeLabel');
    if (icon)  icon.textContent  = dark ?  '☀️' : '🌙';
    if (label) label.textContent = dark ? 'Light mode' : 'Dark mode' ;
}

function toggleTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    applyTheme(!isDark);
    try { localStorage.setItem('pose2sim-theme', !isDark ? 'dark' : 'light'); } catch(e) {}
}

// Restore saved theme on page load
(function () {
    try {
        const saved = localStorage.getItem('pose2sim-theme');
        if (saved === 'dark') applyTheme(true);
        else if (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches) applyTheme(true);
    } catch(e) {}
})();

// ---------------------------------------------------------------------------
// Mobile sidebar (hamburger)
// ---------------------------------------------------------------------------
function toggleSidebar() {
    const sidebar  = document.getElementById('sidebar');
    const btn      = document.getElementById('hamburgerBtn');
    const overlay  = document.getElementById('sidebarOverlay');
    const isOpen   = sidebar.classList.toggle('mobile-open');
    btn.classList.toggle('open', isOpen);
    overlay.classList.toggle('visible', isOpen);
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const btn     = document.getElementById('hamburgerBtn');
    const overlay = document.getElementById('sidebarOverlay');
    sidebar.classList.remove('mobile-open');
    btn.classList.remove('open');
    overlay.classList.remove('visible');
}

// Close sidebar after a nav-item click on mobile
document.addEventListener('click', e => {
    if (e.target.closest('.nav-item') && window.innerWidth <= 768) {
        closeSidebar();
    }
}, true);

// ---------------------------------------------------------------------------
// Keyboard navigation
// ---------------------------------------------------------------------------
document.addEventListener('keydown', e => {
    if (viewAllMode) return;
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   previousSection();
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') nextSection();
});
