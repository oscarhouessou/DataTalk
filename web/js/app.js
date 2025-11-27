// ========================================
// Configuration et État de l'Application
// ========================================

// Détection automatique de l'environnement (local vs déployé)
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:8000' 
    : window.location.origin;
let sessionId = null;
let chatHistory = [];

// ========================================
// Initialisation
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Générer un ID de session unique
    sessionId = generateSessionId();

    // Initialiser les event listeners
    setupEventListeners();
    
    // Charger les suggestions par défaut
    loadSuggestions();

    console.log('DataTalk initialized with session:', sessionId);
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// ========================================
// Event Listeners
// ========================================

function setupEventListeners() {
    // Upload zone
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');

    // Click to browse - empêcher la propagation pour éviter le double déclenchement
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Empêcher la propagation vers dropZone
        fileInput.click();
    });

    dropZone.addEventListener('click', (e) => {
        // Ne déclencher que si on clique directement sur la dropZone, pas sur le bouton
        if (e.target === dropZone || e.target.closest('.drop-zone-content')) {
            fileInput.click();
        }
    });

    // File input change - upload automatique dès la sélection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
            // Réinitialiser l'input pour permettre de sélectionner le même fichier à nouveau
            e.target.value = '';
        }
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');

        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // Chat input
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    sendBtn.addEventListener('click', () => sendMessage());
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Clear chat button
    const clearChatBtn = document.getElementById('clear-chat-btn');
    clearChatBtn.addEventListener('click', clearChat);

    // New file button
    const newFileBtn = document.getElementById('new-file-btn');
    newFileBtn.addEventListener('click', resetApp);
}

// ========================================
// File Upload
// ========================================

async function handleFileUpload(file) {
    // Vérifier le type de fichier
    const validTypes = ['.csv', '.xlsx', '.xls'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(fileExt)) {
        showToast('Format de fichier non supporté. Utilisez CSV ou Excel.', 'error');
        return;
    }

    // Afficher la progression
    showUploadProgress();

    // Créer le FormData
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Erreur lors du téléversement');
        }

        const data = await response.json();

        // Masquer l'upload section et afficher l'app
        hideUploadProgress();
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('app-section').classList.remove('hidden');

        // Mettre à jour les informations du dataset
        updateDatasetInfo(data);

        // Charger les insights et suggestions
        loadInsights();
        loadSuggestions();

        showToast('Fichier chargé avec succès !', 'success');

    } catch (error) {
        hideUploadProgress();
        showToast('Erreur lors du chargement du fichier: ' + error.message, 'error');
        console.error('Upload error:', error);
    }
}

function showUploadProgress() {
    document.getElementById('drop-zone').classList.add('hidden');
    document.getElementById('upload-progress').classList.remove('hidden');

    // Animer la barre de progression
    const progressFill = document.querySelector('.progress-fill');
    progressFill.style.width = '0%';

    setTimeout(() => {
        progressFill.style.width = '90%';
    }, 100);
}

function hideUploadProgress() {
    const progressFill = document.querySelector('.progress-fill');
    progressFill.style.width = '100%';

    setTimeout(() => {
        document.getElementById('upload-progress').classList.add('hidden');
        document.getElementById('drop-zone').classList.remove('hidden');
        progressFill.style.width = '0%';
    }, 500);
}

function updateDatasetInfo(data) {
    document.getElementById('file-name').textContent = data.filename;
    document.getElementById('file-rows').textContent = data.rows.toLocaleString();
    document.getElementById('file-cols').textContent = data.columns;
}

// ========================================
// Insights et Suggestions
// ========================================

async function loadInsights() {
    const insightsContent = document.getElementById('insights-content');
    insightsContent.innerHTML = '<div class="loading-spinner"></div><p class="loading-text">Génération d\'insights...</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/insights`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });

        if (!response.ok) {
            throw new Error('Erreur lors du chargement des insights');
        }

        const data = await response.json();

        if (data.success && data.insights) {
            // Formater les insights pour une meilleure lisibilité
            let formattedInsights = data.insights
                // Convertir les bullet points • en HTML
                .replace(/•\s*Insight\s*\d+:\s*\*\*([^*]+)\*\*/g, '<div style="margin-bottom: 1rem;"><strong style="color: var(--primary); display: block; margin-bottom: 0.25rem;">💡 $1</strong>')
                .replace(/\n\n/g, '</div>')
                // Convertir les ** en strong
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                // Convertir les retours à la ligne simples en <br>
                .replace(/\n/g, '<br>');

            insightsContent.innerHTML = `<div style="line-height: 1.7;">${formattedInsights}</div>`;
        } else {
            insightsContent.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">Aucun insight disponible</p>';
        }

    } catch (error) {
        insightsContent.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">Erreur lors du chargement des insights</p>';
        console.error('Insights error:', error);
    }
}

async function loadSuggestions() {
    const suggestionsContent = document.getElementById('suggestions-content');
    suggestionsContent.innerHTML = '<div class="loading-spinner"></div><p class="loading-text">Génération de suggestions...</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/questions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });

        if (!response.ok) {
            throw new Error('Erreur lors du chargement des suggestions');
        }

        const data = await response.json();

        if (data.success && data.questions && data.questions.length > 0) {
            suggestionsContent.innerHTML = '';
            data.questions.forEach((question, index) => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.textContent = `❓ ${question}`;
                btn.addEventListener('click', () => {
                    document.getElementById('chat-input').value = question;
                    sendMessage();
                });
                suggestionsContent.appendChild(btn);
            });
        } else {
            suggestionsContent.innerHTML = '<p class="text-muted">Aucune suggestion disponible</p>';
        }

    } catch (error) {
        suggestionsContent.innerHTML = '<p class="text-muted">Erreur lors du chargement des suggestions</p>';
        console.error('Suggestions error:', error);
    }
}

// ========================================
// Chat
// ========================================

async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();

    if (!message) return;

    // Ajouter le message de l'utilisateur
    addMessageToChat('user', message);
    chatInput.value = '';

    // Afficher l'indicateur de saisie
    showTypingIndicator();

    try {
        // Envoyer la requête à l'API
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                query: message
            })
        });

        if (!response.ok) {
            throw new Error('Erreur lors de l\'envoi du message');
        }

        const data = await response.json();

        // Masquer l'indicateur de saisie
        hideTypingIndicator();

        // Ajouter la réponse de l'assistant
        addMessageToChat('assistant', data.answer);

        // Vérifier si un graphique est nécessaire
        await checkAndGenerateChart(message, data.answer);

        // Sauvegarder dans l'historique
        chatHistory.push({
            question: message,
            answer: data.answer
        });

    } catch (error) {
        hideTypingIndicator();
        addMessageToChat('assistant', 'Désolé, une erreur s\'est produite: ' + error.message);
        console.error('Chat error:', error);
    }
}

function addMessageToChat(role, text, chartData = null) {
    const chatMessages = document.getElementById('chat-messages');

    // Supprimer le message de bienvenue si présent
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    // Créer le message
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = text;

    content.appendChild(textDiv);

    // Ajouter le graphique si présent
    if (chartData) {
        const chartDiv = document.createElement('div');
        chartDiv.className = 'message-chart';
        const img = document.createElement('img');
        img.src = chartData;
        img.alt = 'Graphique';
        chartDiv.appendChild(img);
        content.appendChild(chartDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatMessages.appendChild(messageDiv);

    // Scroller vers le bas
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');

    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing-indicator-message';
    typingDiv.id = 'typing-indicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

    content.appendChild(typingIndicator);
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(content);

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

async function checkAndGenerateChart(question, answer) {
    try {
        const response = await fetch(`${API_BASE_URL}/chart`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                question: question,
                answer: answer
            })
        });

        if (!response.ok) {
            return;
        }

        const data = await response.json();

        if (data.success && data.has_chart && data.chart_data) {
            // Ajouter le graphique au dernier message de l'assistant
            const messages = document.querySelectorAll('.message.assistant');
            const lastMessage = messages[messages.length - 1];

            if (lastMessage) {
                const content = lastMessage.querySelector('.message-content');
                const chartDiv = document.createElement('div');
                chartDiv.className = 'message-chart';
                const img = document.createElement('img');
                img.src = data.chart_data;
                img.alt = 'Graphique';
                chartDiv.appendChild(img);
                content.appendChild(chartDiv);
            }
        }

    } catch (error) {
        console.error('Chart generation error:', error);
    }
}

function clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">👋</div>
            <h3>Bienvenue sur DataTalk !</h3>
            <p>Posez vos questions sur les données en langage naturel.</p>
            <p class="welcome-hint">Essayez les questions suggérées ou posez votre propre question.</p>
        </div>
    `;
    chatHistory = [];
    showToast('Historique effacé', 'info');
}

// ========================================
// Reset App
// ========================================

function resetApp() {
    // Réinitialiser l'état
    sessionId = generateSessionId();
    chatHistory = [];

    // Masquer l'app section et afficher l'upload section
    document.getElementById('app-section').classList.add('hidden');
    document.getElementById('upload-section').classList.remove('hidden');

    // Réinitialiser le chat
    clearChat();

    // Réinitialiser le file input
    document.getElementById('file-input').value = '';

    showToast('Prêt pour un nouveau fichier', 'info');
}

// ========================================
// Toast Notifications
// ========================================

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    toastContainer.appendChild(toast);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'toastSlideIn 0.3s ease reverse';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}

// ========================================
// Utility Functions
// ========================================

function formatNumber(num) {
    return num.toLocaleString('fr-FR');
}

// ========================================
// Error Handling
// ========================================

window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});
