document.addEventListener('DOMContentLoaded', () => {
    const articleText = document.getElementById('articleText');
    const articleFile = document.getElementById('articleFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const tabTextBtn = document.getElementById('tabTextBtn');
    const tabFileBtn = document.getElementById('tabFileBtn');
    const textInputContainer = document.getElementById('textInputContainer');
    const fileInputContainer = document.getElementById('fileInputContainer');
    const selectedFileName = document.getElementById('selectedFileName');
    const charCountLabel = document.getElementById('charCountLabel');
    const wordCountLabel = document.getElementById('wordCountLabel');

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultsContainer = document.getElementById('resultsContainer');

    const verdictBanner = document.getElementById('verdictBanner');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictSubtitle = document.getElementById('verdictSubtitle');
    const confidenceBar = document.getElementById('confidenceBar');
    const probFakeVal = document.getElementById('probFakeVal');
    const probRealVal = document.getElementById('probRealVal');
    const metricSensational = document.getElementById('metricSensational');
    const metricCaps = document.getElementById('metricCaps');
    const metricPunct = document.getElementById('metricPunct');
    const keyFactorsBadges = document.getElementById('keyFactorsBadges');
    const diagnosticNotes = document.getElementById('diagnosticNotes');
    const timestampBadge = document.getElementById('timestampBadge');

    const historyBody = document.getElementById('historyBody');
    const emptyHistoryRow = document.getElementById('emptyHistoryRow');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');

    // Tab Switching
    tabTextBtn.addEventListener('click', () => {
        tabTextBtn.classList.add('active');
        tabFileBtn.classList.remove('active');
        textInputContainer.classList.remove('d-none');
        fileInputContainer.classList.add('d-none');
    });

    tabFileBtn.addEventListener('click', () => {
        tabFileBtn.classList.add('active');
        tabTextBtn.classList.remove('active');
        textInputContainer.classList.add('d-none');
        fileInputContainer.classList.remove('d-none');
        fileInputContainer.classList.add('d-flex');
    });

    // File selection display
    articleFile.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFileName.textContent = `Selected: ${e.target.files[0].name} (${(e.target.files[0].size / 1024).toFixed(1)} KB)`;
        } else {
            selectedFileName.textContent = '';
        }
    });

    // Word and character live counter
    articleText.addEventListener('input', () => {
        const text = articleText.value.trim();
        const chars = articleText.value.length;
        const words = text ? text.split(/\s+/).length : 0;
        charCountLabel.textContent = `${chars} characters`;
        wordCountLabel.textContent = `${words} words`;
    });

    // Clear input
    clearBtn.addEventListener('click', () => {
        articleText.value = '';
        articleFile.value = '';
        selectedFileName.textContent = '';
        charCountLabel.textContent = '0 characters';
        wordCountLabel.textContent = '0 words';
    });

    // Quick Sample Loader
    let cachedSamples = {};
    fetch('/samples')
        .then(res => res.json())
        .then(samples => {
            samples.forEach(s => {
                cachedSamples[s.id] = s;
            });
        })
        .catch(err => console.error('Failed to load samples:', err));

    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const sid = btn.getAttribute('data-sample-id');
            if (cachedSamples[sid]) {
                tabTextBtn.click();
                articleText.value = cachedSamples[sid].content;
                articleText.dispatchEvent(new Event('input'));
            }
        });
    });

    // Prediction trigger
    analyzeBtn.addEventListener('click', async () => {
        const isFileTab = tabFileBtn.classList.contains('active');
        const formData = new FormData();

        if (isFileTab) {
            if (!articleFile.files || articleFile.files.length === 0) {
                alert('Please select a .txt or .pdf document first.');
                return;
            }
            formData.append('articleFile', articleFile.files[0]);
        } else {
            const text = articleText.value.trim();
            if (!text) {
                alert('Please enter or paste article text to analyze.');
                return;
            }
            formData.append('articleText', text);
        }

        // Show loading state
        emptyState.classList.add('d-none');
        resultsContainer.classList.add('d-none');
        loadingState.classList.remove('d-none');

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            loadingState.classList.add('d-none');

            if (data.status === 'success') {
                renderResults(data);
                addHistoryRecord(data);
            } else {
                alert(data.message || 'An error occurred during prediction.');
                emptyState.classList.remove('d-none');
            }
        } catch (err) {
            loadingState.classList.add('d-none');
            emptyState.classList.remove('d-none');
            alert('Failed to connect to verification backend.');
            console.error(err);
        }
    });

    function renderResults(data) {
        resultsContainer.classList.remove('d-none');
        timestampBadge.textContent = new Date().toLocaleTimeString();

        const isReal = data.verdict === 'REAL';

        verdictBanner.className = `p-4 rounded-3 text-center border ${isReal ? 'verdict-real' : 'verdict-fake'}`;
        verdictIcon.className = `fa-solid ${isReal ? 'fa-circle-check' : 'fa-triangle-exclamation'} fs-2`;
        verdictTitle.textContent = isReal ? 'AUTHENTIC JOURNALISM' : 'UNVERIFIED / FAKE NEWS';
        verdictSubtitle.textContent = isReal 
            ? 'High vocabulary fidelity and journalistic indicators present.'
            : 'Sensationalism or unverified conspiracy markers detected.';

        confidenceBar.style.width = `${data.confidence}%`;
        confidenceBar.textContent = `${data.confidence}% Confidence`;

        probFakeVal.textContent = `${data.probabilities.fake}%`;
        probRealVal.textContent = `${data.probabilities.real}%`;

        metricSensational.textContent = `${data.linguistics.sensational_score}/100`;
        metricCaps.textContent = `${data.linguistics.uppercase_ratio}%`;
        metricPunct.textContent = data.linguistics.exclamation_count;

        // Key Factors
        keyFactorsBadges.innerHTML = '';
        if (data.key_factors && data.key_factors.length > 0) {
            data.key_factors.forEach(f => {
                const badge = document.createElement('span');
                const isFReal = f.supports === 'REAL';
                badge.className = `badge rounded-pill px-2 py-1 ${isFReal ? 'badge-tag-real' : 'badge-tag-fake'}`;
                badge.innerHTML = `<strong>${f.term}</strong> <small>(${f.weight > 0 ? '+' : ''}${f.weight})</small>`;
                keyFactorsBadges.appendChild(badge);
            });
        } else {
            keyFactorsBadges.innerHTML = '<span class="text-secondary small">No specific dominant vocabulary weights isolated.</span>';
        }

        // Diagnostic notes
        diagnosticNotes.innerHTML = '';
        const notes = [];
        if (data.linguistics.sensational_score > 30) {
            notes.push(`⚠️ Elevated sensationalism score (${data.linguistics.sensational_score}/100) typical of clickbait headlines.`);
        }
        if (data.linguistics.sensational_terms_found.length > 0) {
            notes.push(`🚩 Sensational keywords: <em>${data.linguistics.sensational_terms_found.join(', ')}</em>`);
        }
        if (data.linguistics.credible_terms_found.length > 0) {
            notes.push(`✅ Credible journalistic terms: <em>${data.linguistics.credible_terms_found.join(', ')}</em>`);
        }
        if (data.linguistics.uppercase_ratio > 15) {
            notes.push(`⚠️ Excessive uppercase capitalization (${data.linguistics.uppercase_ratio}% of words) indicates emotional emphasis.`);
        }
        if (notes.length === 0) {
            notes.push('ℹ️ Standard linguistic baseline structure with normal punctuation distribution.');
        }

        notes.forEach(n => {
            const p = document.createElement('div');
            p.innerHTML = n;
            diagnosticNotes.appendChild(p);
        });
    }

    function addHistoryRecord(data) {
        if (emptyHistoryRow) {
            emptyHistoryRow.remove();
        }

        const isReal = data.verdict === 'REAL';
        const row = document.createElement('tr');
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const snippet = data.preview || 'Text snippet';

        row.innerHTML = `
            <td class="ps-4 text-secondary">${timeStr}</td>
            <td><span class="badge ${isReal ? 'bg-success' : 'bg-danger'}">${data.verdict}</span></td>
            <td><strong>${data.confidence}%</strong></td>
            <td><span class="badge bg-dark border border-secondary">${data.linguistics.sensational_score}/100</span></td>
            <td class="text-truncate text-secondary" style="max-width: 260px;">${snippet}</td>
            <td class="pe-4 text-end">
                <button class="btn btn-sm btn-outline-primary py-0 px-2 view-again-btn">Inspect</button>
            </td>
        `;

        row.querySelector('.view-again-btn').addEventListener('click', () => {
            renderResults(data);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });

        historyBody.insertBefore(row, historyBody.firstChild);
    }

    clearHistoryBtn.addEventListener('click', () => {
        historyBody.innerHTML = `
            <tr id="emptyHistoryRow">
                <td colspan="6" class="text-center py-4 text-secondary">
                    No articles analyzed in this session yet.
                </td>
            </tr>
        `;
    });
});
