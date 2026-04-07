/* BallCut - Shared JS utilities */

function showError(elementId, msg) {
    const el = document.getElementById(elementId);
    if (el) el.textContent = msg;
}

function clearError(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.textContent = '';
}
