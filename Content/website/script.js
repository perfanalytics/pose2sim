'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let sections    = [];   // Array of section IDs in DOM order
let currentIdx  = 0;    // Index of currently visible section
let viewAllMode = false;

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
    showSection(targetSection, false);
    updateNavButtons();
    updateActiveNav(targetSection);

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
    showSection(targetSection, false);
    // Scroll to sub-anchor if it's not the section itself
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
    return sections[0];
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function goToSection(sectionId, anchor) {
    if (viewAllMode) toggleViewAll();
    showSection(sectionId, !anchor);
    updateNavButtons();
    updateActiveNav(sectionId);
    if (anchor) {
        const el = document.getElementById(anchor);
        if (el) setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
    }
}

function showSection(sectionId, scrollTop = true) {
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    const el = document.getElementById(sectionId);
    if (el) {
        el.classList.add('active');
        currentIdx = sections.indexOf(sectionId);
        history.replaceState(null, '', `#${sectionId}`);
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
    nextBtn.querySelector('span').textContent =
        currentIdx === sections.length - 1 ? 'Finish ✓' : 'Next →';
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
// Keyboard navigation
// ---------------------------------------------------------------------------
document.addEventListener('keydown', e => {
    if (viewAllMode) return;
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   previousSection();
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') nextSection();
});
