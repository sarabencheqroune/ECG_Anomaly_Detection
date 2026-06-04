#!/usr/bin/env python
"""
Main Flask application for ECG Anomaly Detection Dashboard
Run with: python -m src.dashboard.app
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask
from flask_socketio import SocketIO

# Create application
app = Flask(__name__)

# SocketIO initialized in __init__.py with proper CORS config

from . import create_app
from .socket_handlers import register_socket_handlers

# Create application using factory
app = create_app()

# Register socket handlers
register_socket_handlers(socketio)

if __name__ == '__main__':
    print("=" * 60)
    print("ECG Anomaly Detection Dashboard")
    print("=" * 60)
    print("Starting server...")
    print("Access the dashboard at: http://localhost:5001")
    print("WebSocket endpoint: ws://localhost:5001/socket.io")
    print("=" * 60)
    
    # Run with SocketIO support - using threading instead of eventlet
    import os
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)