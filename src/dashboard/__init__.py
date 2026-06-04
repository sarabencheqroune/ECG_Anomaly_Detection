"""
Flask Dashboard for ECG Anomaly Detection
Real-time visualization and monitoring of ECG predictions
"""

from flask import Flask
from flask_socketio import SocketIO
from pathlib import Path

import os

# Initialize SocketIO globally with secure CORS
allowed_origins = os.environ.get('CORS_ALLOWED_ORIGINS', 'http://localhost:5001').split(',')
socketio = SocketIO(cors_allowed_origins=allowed_origins, async_mode='threading')

def create_app(config=None):
    """Application factory pattern"""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    
    if config:
        app.config.update(config)
    
    # Initialize extensions
    socketio.init_app(app)
    
    # Register blueprints
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app

__all__ = ['create_app', 'socketio']