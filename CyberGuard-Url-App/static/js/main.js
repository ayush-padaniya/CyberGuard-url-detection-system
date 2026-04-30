// Scanning animation
document.querySelector('.url-form')?.addEventListener('submit', function() {
    const btn = document.querySelector('.scan-btn');
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning...';
    btn.disabled = true;
});

document.querySelectorAll('.confidence-fill').forEach(function(fill) {
    const confidence = Number(fill.dataset.confidence || 0);
    fill.style.width = `${Math.max(0, Math.min(confidence, 100))}%`;
});

// Auto focus input
document.querySelector('input[name="url"]')?.focus();
