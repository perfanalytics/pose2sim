// Global variables
let currentStep = 0;
const totalSteps = 16; // FIXED: Changed from 11 to 16 (steps 0-15)
let currentLanguage = 'en';
let viewAllMode = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    updateNavButtons();
    updateActiveNavItem();
});

// Navigation functions
function goToStep(stepNumber) {
    if (viewAllMode) {
        toggleViewAll(); // Exit view all mode
    }
    
    currentStep = stepNumber;
    showStep(currentStep);
    updateNavButtons();
    updateActiveNavItem();
    scrollToTop();
}

function nextStep() {
    if (viewAllMode) {
        toggleViewAll();
        return;
    }
    
    if (currentStep < totalSteps - 1) {
        currentStep++;
        showStep(currentStep);
        updateNavButtons();
        updateActiveNavItem();
        scrollToTop();
    }
}

function previousStep() {
    if (viewAllMode) {
        toggleViewAll();
        return;
    }
    
    if (currentStep > 0) {
        currentStep--;
        showStep(currentStep);
        updateNavButtons();
        updateActiveNavItem();
        scrollToTop();
    }
}

function showStep(stepNumber) {
    // Hide all steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active');
    });
    
    // Show current step
    const currentStepElement = document.getElementById(`step-${stepNumber}`);
    if (currentStepElement) {
        currentStepElement.classList.add('active');
    }
}

function updateNavButtons() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    
    if (viewAllMode) {
        prevBtn.style.display = 'none';
        nextBtn.querySelector('span').textContent = 'Exit View All';
        return;
    }
    
    // Show/hide previous button
    prevBtn.style.display = currentStep === 0 ? 'none' : 'inline-flex';
    
    // Update next button text
    if (currentStep === totalSteps - 1) {
        nextBtn.querySelector('span').textContent = 'Finish ✓';
    } else {
        nextBtn.querySelector('span').textContent = 'Next →';
    }
}

function updateActiveNavItem() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const activeItem = document.querySelector(`a[href="#step-${currentStep}"]`);
    if (activeItem) {
        activeItem.classList.add('active');
    }
}

function toggleViewAll() {
    viewAllMode = !viewAllMode;
    
    const steps = document.querySelectorAll('.step');
    const viewAllBtn = document.querySelector('.view-all-btn');
    
    if (viewAllMode) {
        // Show all steps
        steps.forEach(step => {
            step.classList.add('view-all-mode');
            step.classList.add('active');
        });
        
        viewAllBtn.querySelector('span').textContent = 'Back to Step View';
        
        document.querySelector('.nav-buttons').style.display = 'flex';
    } else {
        // Return to single step view
        steps.forEach(step => {
            step.classList.remove('view-all-mode');
            step.classList.remove('active');
        });
        
        showStep(currentStep);
        
        viewAllBtn.querySelector('span').textContent = 'View All Steps';
    }
    
    updateNavButtons();
    scrollToTop();
}

function initializeNavigation() {
    // Add click handlers to nav items
    document.querySelectorAll('.nav-item').forEach((item, index) => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            goToStep(index);
        });
    });
}

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Keyboard navigation
document.addEventListener('keydown', function(e) {
    if (viewAllMode) return;
    
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        if (currentStep > 0) {
            previousStep();
        }
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        if (currentStep < totalSteps - 1) {
            nextStep();
        }
    }
});