/**
 * WebSocket Client for ECG Dashboard
 * Handles real-time communication with Flask-SocketIO server
 */

class ECGWebSocketClient {
    constructor(options = {}) {
        this.url = options.url || 'ws://localhost:5000';
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
        this.reconnectDelay = options.reconnectDelay || 2000;
        
        // Callbacks
        this.onConnected = options.onConnected || (() => {});
        this.onDisconnected = options.onDisconnected || (() => {});
        this.onECGData = options.onECGData || (() => {});
        this.onPrediction = options.onPrediction || (() => {});
        this.onError = options.onError || (() => {});
        this.onStreamStarted = options.onStreamStarted || (() => {});
        this.onStreamStopped = options.onStreamStopped || (() => {});
        
        this.initSocket();
    }
    
    initSocket() {
        // Load Socket.IO client library
        if (typeof io === 'undefined') {
            console.error('Socket.IO client library not loaded');
            return;
        }
        
        this.socket = io(this.url, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: this.maxReconnectAttempts,
            reconnectionDelay: this.reconnectDelay
        });
        
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.onConnected();
        });
        
        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.onDisconnected();
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('WebSocket connection error:', error);
            this.reconnectAttempts++;
            this.onError(error);
        });
        
        this.socket.on('ecg_data', (data) => {
            this.onECGData(data);
        });
        
        this.socket.on('prediction_result', (data) => {
            this.onPrediction(data);
        });
        
        this.socket.on('stream_started', (data) => {
            console.log('Stream started:', data);
            this.onStreamStarted(data);
        });
        
        this.socket.on('stream_stopped', (data) => {
            console.log('Stream stopped:', data);
            this.onStreamStopped(data);
        });
        
        this.socket.on('stream_ended', (data) => {
            console.log('Stream ended:', data);
            this.onStreamStopped(data);
        });
        
        this.socket.on('feedback_received', (data) => {
            console.log('Feedback received:', data);
        });
        
        this.socket.on('error', (error) => {
            console.error('Server error:', error);
            this.onError(error);
        });
    }
    
    startStream(options = {}) {
        if (!this.isConnected) {
            console.warn('Not connected to server');
            return false;
        }
        
        const streamConfig = {
            sampling_rate: options.samplingRate || 360,
            chunk_size: options.chunkSize || 360,
            duration: options.duration || 0,
            heart_rate: options.heartRate || 75
        };
        
        this.socket.emit('start_stream', streamConfig);
        return true;
    }
    
    stopStream() {
        if (!this.isConnected) return false;
        this.socket.emit('stop_stream');
        return true;
    }
    
    getPrediction(waveform) {
        if (!this.isConnected) {
            console.warn('Not connected to server');
            return false;
        }
        
        this.socket.emit('get_prediction', { waveform: waveform });
        return true;
    }
    
    submitFeedback(feedback) {
        if (!this.isConnected) return false;
        this.socket.emit('review_feedback', feedback);
        return true;
    }
    
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.isConnected = false;
        }
    }
    
    reconnect() {
        if (this.socket) {
            this.socket.connect();
        }
    }
}

// Real-time Dashboard Manager
class DashboardManager {
    constructor() {
        this.wsClient = null;
        this.visualizer = null;
        this.predictionHistory = [];
        this.reviewQueue = [];
        this.chartInstances = {};
        this.settings = this.loadSettings();
        
        this.init();
    }
    
    init() {
        // Initialize WebSocket client
        this.wsClient = new ECGWebSocketClient({
            onConnected: () => this.onConnected(),
            onDisconnected: () => this.onDisconnected(),
            onECGData: (data) => this.onECGData(data),
            onPrediction: (data) => this.onPrediction(data),
            onError: (error) => this.onError(error),
            onStreamStarted: () => this.onStreamStarted(),
            onStreamStopped: () => this.onStreamStopped()
        });
        
        // Initialize visualizer
        this.visualizer = new ECGPlotlyVisualizer('waveform-plot', {
            samplingRate: this.settings.samplingRate || 360,
            durationSeconds: this.settings.durationSeconds || 5,
            lineColor: this.settings.waveformColor || '#00ff00'
        });
        
        // Bind UI events
        this.bindEvents();
        
        // Start periodic updates
        this.startPeriodicUpdates();
        
        console.log('Dashboard initialized');
    }
    
    bindEvents() {
        // Stream control buttons
        document.getElementById('start-stream')?.addEventListener('click', () => {
            this.wsClient.startStream({
                samplingRate: this.settings.samplingRate,
                chunkSize: this.settings.chunkSize,
                heartRate: this.settings.heartRate
            });
        });
        
        document.getElementById('stop-stream')?.addEventListener('click', () => {
            this.wsClient.stopStream();
            this.visualizer?.stopAnimation();
        });
        
        document.getElementById('clear-waveform')?.addEventListener('click', () => {
            this.visualizer?.clear();
        });
        
        document.getElementById('export-waveform')?.addEventListener('click', () => {
            this.visualizer?.exportAsImage();
        });
        
        // Settings save
        document.getElementById('save-settings')?.addEventListener('click', () => {
            this.saveSettings();
        });
        
        // Review feedback
        document.getElementById('submit-feedback')?.addEventListener('click', () => {
            this.submitFeedback();
        });
    }
    
    onConnected() {
        console.log('Connected to server');
        this.updateConnectionStatus(true);
        this.showNotification('Connected to ECG Server', 'success');
    }
    
    onDisconnected() {
        console.log('Disconnected from server');
        this.updateConnectionStatus(false);
        this.showNotification('Disconnected from server', 'error');
    }
    
    onECGData(data) {
        // Update waveform
        if (data.waveform) {
            this.visualizer?.addDataChunk(data.waveform);
        }
        
        // Update prediction display
        if (data.prediction) {
            this.updatePredictionDisplay(data.prediction);
        }
        
        // Update metrics
        this.updateMetrics(data);
    }
    
    onPrediction(data) {
        const prediction = data.prediction;
        this.predictionHistory.unshift({
            timestamp: data.timestamp,
            ...prediction
        });
        
        // Keep last 100 predictions
        if (this.predictionHistory.length > 100) {
            this.predictionHistory.pop();
        }
        
        this.updatePredictionDisplay(prediction);
        this.updatePredictionHistoryTable();
        
        // Check if needs review
        if (prediction.confidence < this.settings.confidenceThreshold) {
            this.addToReviewQueue(prediction);
            this.showNotification(
                `Low confidence prediction: ${prediction.class} (${(prediction.confidence * 100).toFixed(1)}%)`,
                'warning'
            );
        }
    }
    
    onError(error) {
        console.error('WebSocket error:', error);
        this.showNotification(`Error: ${error.message || 'Unknown error'}`, 'error');
    }
    
    onStreamStarted() {
        this.visualizer?.startAnimation();
        this.updateStreamStatus(true);
        this.showNotification('ECG stream started', 'info');
    }
    
    onStreamStopped() {
        this.visualizer?.stopAnimation();
        this.updateStreamStatus(false);
        this.showNotification('ECG stream stopped', 'info');
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? '● Connected' : '○ Disconnected';
            statusElement.className = `status-badge ${connected ? 'connected' : 'disconnected'}`;
        }
    }
    
    updateStreamStatus(active) {
        const indicator = document.getElementById('stream-indicator');
        if (indicator) {
            indicator.textContent = active ? 'Streaming Active' : 'Streaming Inactive';
            indicator.style.color = active ? '#27ae60' : '#e74c3c';
        }
    }
    
    updatePredictionDisplay(prediction) {
        // Update class
        const classElement = document.getElementById('prediction-class');
        if (classElement) {
            classElement.textContent = prediction.class;
            classElement.className = `prediction-class ${prediction.class.toLowerCase()}`;
        }
        
        // Update confidence
        const confidenceElement = document.getElementById('prediction-confidence');
        if (confidenceElement) {
            const confidencePercent = (prediction.confidence * 100).toFixed(1);
            confidenceElement.textContent = `${confidencePercent}% Confidence`;
            
            // Update confidence bar
            const confidenceFill = document.querySelector('.confidence-fill');
            if (confidenceFill) {
                confidenceFill.style.width = `${confidencePercent}%`;
                confidenceFill.className = `confidence-fill ${
                    prediction.confidence < 0.5 ? 'low' : 
                    prediction.confidence < 0.7 ? 'medium' : ''
                }`;
            }
        }
        
        // Update anomaly indicator
        const anomalyIndicator = document.getElementById('anomaly-indicator');
        if (anomalyIndicator && prediction.anomaly_score) {
            const isAnomaly = prediction.anomaly_score > 0.5;
            anomalyIndicator.textContent = isAnomaly ? '⚠ ANOMALY' : '✓ NORMAL';
            anomalyIndicator.className = `anomaly-indicator ${isAnomaly ? 'anomaly' : 'normal'}`;
        }
        
        // Update heart rate display
        if (prediction.heart_rate) {
            const hrElement = document.getElementById('heart-rate-value');
            if (hrElement) {
                hrElement.textContent = Math.round(prediction.heart_rate);
            }
        }
    }
    
    updateMetrics(data) {
        // Update total predictions count
        const totalElement = document.getElementById('total-predictions');
        if (totalElement) {
            totalElement.textContent = this.predictionHistory.length;
        }
        
        // Update average confidence
        if (this.predictionHistory.length > 0) {
            const avgConfidence = this.predictionHistory.reduce((sum, p) => sum + p.confidence, 0) / this.predictionHistory.length;
            const avgElement = document.getElementById('avg-confidence');
            if (avgElement) {
                avgElement.textContent = `${(avgConfidence * 100).toFixed(1)}%`;
            }
        }
        
        // Update class distribution
        this.updateClassDistribution();
    }
    
    updateClassDistribution() {
        const distribution = {};
        this.predictionHistory.forEach(p => {
            distribution[p.class] = (distribution[p.class] || 0) + 1;
        });
        
        const container = document.getElementById('class-distribution');
        if (container) {
            container.innerHTML = '';
            for (const [className, count] of Object.entries(distribution)) {
                const percentage = (count / this.predictionHistory.length * 100).toFixed(1);
                const element = document.createElement('div');
                element.className = 'stat-item';
                element.innerHTML = `
                    <div class="stat-value">${count}</div>
                    <div class="stat-label">${className} (${percentage}%)</div>
                `;
                container.appendChild(element);
            }
        }
    }
    
    updatePredictionHistoryTable() {
        const tableBody = document.getElementById('prediction-history-body');
        if (!tableBody) return;
        
        tableBody.innerHTML = '';
        this.predictionHistory.slice(0, 10).forEach(pred => {
            const row = tableBody.insertRow();
            row.insertCell(0).textContent = new Date(pred.timestamp).toLocaleTimeString();
            row.insertCell(1).textContent = pred.class;
            row.insertCell(2).textContent = `${(pred.confidence * 100).toFixed(1)}%`;
            row.insertCell(3).textContent = pred.anomaly_score ? pred.anomaly_score.toFixed(3) : 'N/A';
            
            const needsReview = pred.confidence < this.settings.confidenceThreshold;
            row.insertCell(4).textContent = needsReview ? '⚠ Yes' : '✓ No';
            row.className = needsReview ? 'review-item warning' : '';
        });
    }
    
    addToReviewQueue(prediction) {
        this.reviewQueue.unshift({
            id: `review_${Date.now()}`,
            timestamp: new Date().toISOString(),
            prediction: prediction,
            reviewed: false
        });
        
        // Keep last 20 review items
        if (this.reviewQueue.length > 20) {
            this.reviewQueue.pop();
        }
        
        this.updateReviewQueue();
    }
    
    updateReviewQueue() {
        const container = document.getElementById('review-queue');
        if (!container) return;
        
        container.innerHTML = '';
        this.reviewQueue.filter(item => !item.reviewed).forEach(item => {
            const priority = item.prediction.confidence < 0.3 ? 'critical' : 
                            item.prediction.confidence < 0.5 ? 'high' :
                            item.prediction.confidence < 0.7 ? 'medium' : 'low';
            
            const div = document.createElement('div');
            div.className = `review-item ${priority}`;
            div.innerHTML = `
                <div><strong>${item.prediction.class}</strong> (${(item.prediction.confidence * 100).toFixed(1)}% confidence)</div>
                <div class="review-actions">
                    <button onclick="dashboardManager.approvePrediction('${item.id}')" class="btn btn-success btn-sm">Approve</button>
                    <button onclick="dashboardManager.correctPrediction('${item.id}')" class="btn btn-warning btn-sm">Correct</button>
                </div>
            `;
            container.appendChild(div);
        });
    }
    
    approvePrediction(itemId) {
        const item = this.reviewQueue.find(i => i.id === itemId);
        if (item) {
            item.reviewed = true;
            this.updateReviewQueue();
            this.showNotification('Prediction approved', 'success');
        }
    }
    
    correctPrediction(itemId) {
        const item = this.reviewQueue.find(i => i.id === itemId);
        if (item) {
            // Open correction modal
            this.openCorrectionModal(item);
        }
    }
    
    openCorrectionModal(item) {
        const modal = document.getElementById('correction-modal');
        if (modal) {
            modal.style.display = 'block';
            document.getElementById('correction-item-id').value = item.id;
            document.getElementById('current-prediction').textContent = item.prediction.class;
        }
    }
    
    submitFeedback() {
        const itemId = document.getElementById('correction-item-id')?.value;
        const correctedLabel = document.getElementById('corrected-label')?.value;
        const reviewerConfidence = parseFloat(document.getElementById('reviewer-confidence')?.value || 1);
        const notes = document.getElementById('review-notes')?.value;
        
        if (itemId && correctedLabel) {
            this.wsClient.submitFeedback({
                item_id: itemId,
                corrected_label: correctedLabel,
                confidence: reviewerConfidence,
                notes: notes
            });
            
            // Mark as reviewed
            const item = this.reviewQueue.find(i => i.id === itemId);
            if (item) {
                item.reviewed = true;
                this.updateReviewQueue();
            }
            
            // Close modal
            this.closeCorrectionModal();
            this.showNotification('Feedback submitted', 'success');
        }
    }
    
    closeCorrectionModal() {
        const modal = document.getElementById('correction-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    loadSettings() {
        const defaultSettings = {
            confidenceThreshold: 0.7,
            samplingRate: 360,
            chunkSize: 360,
            durationSeconds: 5,
            waveformColor: '#00ff00',
            heartRate: 75,
            showAnomalyScore: true,
            alertOnLowConfidence: true
        };
        
        const saved = localStorage.getItem('ecg_dashboard_settings');
        return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
    }
    
    saveSettings() {
        this.settings = {
            confidenceThreshold: parseFloat(document.getElementById('conf-threshold')?.value || 0.7),
            samplingRate: parseInt(document.getElementById('sampling-rate')?.value || 360),
            durationSeconds: parseInt(document.getElementById('duration')?.value || 5),
            waveformColor: document.getElementById('waveform-color')?.value || '#00ff00',
            showAnomalyScore: document.getElementById('show-anomaly')?.checked || false,
            alertOnLowConfidence: document.getElementById('alert-low-conf')?.checked || true
        };
        
        localStorage.setItem('ecg_dashboard_settings', JSON.stringify(this.settings));
        
        // Apply settings
        this.visualizer?.setDuration(this.settings.durationSeconds);
        this.visualizer?.setColor(this.settings.waveformColor);
        
        this.showNotification('Settings saved', 'success');
    }
    
    startPeriodicUpdates() {
        // Fetch statistics every 5 seconds
        setInterval(() => {
            this.fetchStats();
        }, 5000);
    }
    
    async fetchStats() {
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            this.updateStatsDisplay(stats);
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    }
    
    updateStatsDisplay(stats) {
        const reviewCountElement = document.getElementById('needs-review-count');
        if (reviewCountElement) {
            reviewCountElement.textContent = stats.needs_review_count || 0;
        }
    }
    
    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `toast-notification alert alert-${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()" style="background:none;border:none;float:right;cursor:pointer;">&times;</button>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize dashboard when page loads
let dashboardManager = null;

document.addEventListener('DOMContentLoaded', () => {
    dashboardManager = new DashboardManager();
    
    // Make available globally for onclick handlers
    window.dashboardManager = dashboardManager;
});