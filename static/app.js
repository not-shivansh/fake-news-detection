const textarea = document.getElementById('newsInput');
const btnAnalyze = document.getElementById('btnAnalyze');
const charCounter = document.getElementById('charCounter');
const resultSec = document.getElementById('resultSection');

textarea.addEventListener('input', updateCounter);

function updateCounter() {
    const words = textarea.value.trim().split(/\s+/).filter(Boolean).length;
    const chars = textarea.value.length;
    charCounter.textContent = words ? `${words.toLocaleString()} words · ${chars.toLocaleString()} chars` : '0 words';
}

async function analyze() {
    const text = textarea.value.trim();
    if (!text) { showToast('Please paste a news article first.'); return; }
    if (text.split(/\s+/).length < 5) { showToast('Text is too short. Please enter more content.'); return; }

    setLoading(true);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || 'Prediction failed. Please try again.');
            return;
        }

        showResult(data);

    } catch (err) {
        showToast('Network error. Is the server running?');
        console.error(err);
    } finally {
        setLoading(false);
    }
}

function showResult(data) {
    const isFake = data.label === 'Fake';

    const card = document.getElementById('resultCard');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictText = document.getElementById('verdictText');
    const verdictSub = document.getElementById('verdictSubtitle');
    const confValue = document.getElementById('confidenceValue');
    const confBar = document.getElementById('confidenceBarFill');
    const statWords = document.getElementById('statWords');
    const statChars = document.getElementById('statChars');
    const statModel = document.getElementById('statModel');

    card.className = 'card result-card ' + (isFake ? 'fake' : 'real');

    verdictIcon.textContent = isFake ? '🚨' : '✅';
    verdictText.textContent = isFake ? 'FAKE NEWS' : 'REAL NEWS';
    verdictSub.textContent = isFake
        ? `Our model is ${data.confidence}% confident this is misinformation.`
        : `Our model is ${data.confidence}% confident this appears to be credible.`;

    confValue.textContent = `${data.confidence}%`;
    confBar.style.width = '0%';

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            confBar.style.width = `${data.confidence}%`;
        });
    });

    statWords.textContent = data.word_count?.toLocaleString() ?? '—';
    statChars.textContent = data.char_count?.toLocaleString() ?? '—';
    statModel.textContent = 'ML';

    resultSec.style.display = 'block';
    resultSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setLoading(yes) {
    btnAnalyze.disabled = yes;
    btnAnalyze.classList.toggle('loading', yes);
}

function clearInput() {
    textarea.value = '';
    updateCounter();
    resultSec.style.display = 'none';
    textarea.focus();
}

function focusInput() {
    resultSec.style.display = 'none';
    textarea.focus();
    textarea.scrollIntoView({ behavior: 'smooth' });
}

let toastTimer;
function showToast(msg) {
    const toast = document.getElementById('errorToast');
    document.getElementById('toastMsg').textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
}

textarea.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze();
});
