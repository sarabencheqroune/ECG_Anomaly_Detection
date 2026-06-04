"""
API routes for the ECG dashboard
"""

from flask import Blueprint, render_template, request, jsonify, send_from_directory
from flask_socketio import emit
import json
from datetime import datetime
import numpy as np
import os

# Create blueprint
main_bp = Blueprint('main', __name__)

# Store for recent predictions
recent_predictions = []
MAX_PREDICTIONS = 100

# Mock model for demonstration (replace with actual model in production)
class MockModel:
    def predict(self, waveform):
        import numpy as np
        # Simple rule-based "prediction" for demo
        std = np.std(waveform)
        if std < 0.1:
            return {'class': 'Normal', 'confidence': 0.92, 'anomaly_score': 0.05}
        elif std > 0.3:
            return {'class': 'PVC', 'confidence': 0.78, 'anomaly_score': 0.45}
        else:
            return {'class': 'AFib', 'confidence': 0.65, 'anomaly_score': 0.32}

model = MockModel()


@main_bp.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@main_bp.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@main_bp.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction on uploaded waveform"""
    try:
        data = request.get_json()
        waveform = np.array(data.get('waveform', []))
        
        if len(waveform) == 0:
            return jsonify({'error': 'No waveform data provided'}), 400
        
        # Get prediction
        prediction = model.predict(waveform)
        
        # Store prediction
        prediction['timestamp'] = datetime.now().isoformat()
        recent_predictions.append(prediction)
        
        # Keep only recent predictions
        if len(recent_predictions) > MAX_PREDICTIONS:
            recent_predictions.pop(0)
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/api/stats')
def get_stats():
    """Get dashboard statistics"""
    if not recent_predictions:
        return jsonify({
            'total_predictions': 0,
            'class_distribution': {},
            'avg_confidence': 0,
            'needs_review_count': 0
        })
    
    # Calculate statistics
    class_dist = {}
    total_confidence = 0
    needs_review = 0
    
    for pred in recent_predictions:
        class_name = pred.get('class', 'Unknown')
        class_dist[class_name] = class_dist.get(class_name, 0) + 1
        total_confidence += pred.get('confidence', 0)
        if pred.get('confidence', 1) < 0.7:
            needs_review += 1
    
    return jsonify({
        'total_predictions': len(recent_predictions),
        'class_distribution': class_dist,
        'avg_confidence': total_confidence / len(recent_predictions),
        'needs_review_count': needs_review,
        'timestamp': datetime.now().isoformat()
    })


@main_bp.route('/api/predictions/history')
def get_prediction_history():
    """Get recent prediction history"""
    return jsonify({
        'predictions': recent_predictions[-50:],  # Last 50 predictions
        'count': len(recent_predictions)
    })


@main_bp.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


@main_bp.route('/api/simulate', methods=['POST'])
def simulate_ecg():
    """Simulate ECG data stream for testing"""
    import numpy as np
    
    data = request.get_json()
    duration = data.get('duration', 5)  # seconds
    sampling_rate = data.get('sampling_rate', 360)
    heart_rate = data.get('heart_rate', 75)
    
    # Generate synthetic ECG
    t = np.linspace(0, duration, int(duration * sampling_rate))
    ecg = np.sin(2 * np.pi * 1.2 * t) * 0.1
    
    # Add R-peaks
    beat_interval = 60 / heart_rate
    for i in range(0, len(t), int(beat_interval * sampling_rate)):
        if i + 30 < len(t):
            ecg[i:i+30] += np.hanning(30) * 0.8
    
    # Add noise
    ecg += 0.05 * np.random.randn(len(t))
    
    return jsonify({
        'waveform': ecg.tolist(),
        'sampling_rate': sampling_rate,
        'duration': duration,
        'heart_rate': heart_rate
    })


@main_bp.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update dashboard settings"""
    settings_file = Path(__file__).parent / 'static' / 'settings.json'
    
    if request.method == 'GET':
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({
                'confidence_threshold': 0.7,
                'update_interval_ms': 500,
                'show_anomaly_score': True,
                'alert_on_low_confidence': True,
                'heartbeat_color': '#00ff00',
                'anomaly_color': '#ff0000'
            })
    
    elif request.method == 'POST':
        settings = request.get_json()
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        return jsonify({'status': 'saved'})