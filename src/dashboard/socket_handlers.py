"""
WebSocket handlers for real-time ECG streaming
"""

from flask import request
from flask_socketio import emit, disconnect
import numpy as np
from datetime import datetime
import time
import threading

# Store active connections
active_connections = {}
streaming_threads = {}


def register_socket_handlers(socketio):
    """Register all socket event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        client_id = request.sid
        active_connections[client_id] = {
            'connected_at': datetime.now().isoformat(),
            'streaming': False
        }
        emit('connected', {
            'message': 'Connected to ECG dashboard',
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        })
        print(f"Client connected: {client_id}")
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        client_id = request.sid
        if client_id in active_connections:
            # Stop any active streaming
            if client_id in streaming_threads:
                streaming_threads[client_id].stop()
                del streaming_threads[client_id]
            del active_connections[client_id]
        print(f"Client disconnected: {client_id}")
    
    @socketio.on('start_stream')
    def handle_start_stream(data):
        """Start streaming ECG data to client"""
        client_id = request.sid
        
        if client_id not in active_connections:
            emit('error', {'message': 'Not connected'})
            return
        
        # Get stream parameters
        sampling_rate = data.get('sampling_rate', 360)
        chunk_size = data.get('chunk_size', 360)  # 1 second chunks
        duration = data.get('duration', 0)  # 0 = infinite
        
        # Stop existing stream if any
        if client_id in streaming_threads:
            streaming_threads[client_id].stop()
        
        # Start new stream
        streamer = ECGStreamer(socketio, client_id, sampling_rate, chunk_size, duration)
        streaming_threads[client_id] = streamer
        streamer.start()
        
        emit('stream_started', {
            'message': 'Streaming started',
            'sampling_rate': sampling_rate,
            'chunk_size': chunk_size
        })
    
    @socketio.on('stop_stream')
    def handle_stop_stream():
        """Stop streaming ECG data"""
        client_id = request.sid
        
        if client_id in streaming_threads:
            streaming_threads[client_id].stop()
            del streaming_threads[client_id]
            emit('stream_stopped', {'message': 'Streaming stopped'})
    
    @socketio.on('get_prediction')
    def handle_prediction_request(data):
        """Get prediction for a specific waveform"""
        waveform = data.get('waveform', [])
        
        if not waveform:
            emit('prediction_error', {'error': 'No waveform provided'})
            return
        
        # Run prediction
        from .routes import model
        prediction = model.predict(np.array(waveform))
        
        emit('prediction_result', {
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
    
    @socketio.on('review_feedback')
    def handle_review_feedback(data):
        """Handle expert review feedback"""
        item_id = data.get('item_id')
        corrected_label = data.get('corrected_label')
        reviewer_confidence = data.get('confidence', 1.0)
        notes = data.get('notes', '')
        
        # Store feedback (in production, save to database)
        feedback = {
            'item_id': item_id,
            'corrected_label': corrected_label,
            'reviewer_confidence': reviewer_confidence,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log feedback
        print(f"Review feedback received: {feedback}")
        
        emit('feedback_received', {
            'message': 'Feedback recorded',
            'feedback': feedback
        })


class ECGStreamer:
    """Simulate real-time ECG stream for WebSocket clients"""
    
    def __init__(self, socketio, client_id, sampling_rate=360, chunk_size=360, duration=0):
        self.socketio = socketio
        self.client_id = client_id
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.duration = duration
        self.running = False
        self.thread = None
        self.time = 0
        self.beat_counter = 0
        self.current_beat = np.zeros(int(0.8 * sampling_rate))  # Initialize with default beat
        
    def start(self):
        """Start the streaming thread"""
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the streaming thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            
    def _generate_ecg_chunk(self):
        """Generate a chunk of synthetic ECG data"""
        duration = self.chunk_size / self.sampling_rate
        t = np.linspace(0, duration, self.chunk_size)
        
        # Generate ECG waveform
        ecg = np.zeros(self.chunk_size)
        
        # Add baseline
        ecg += 0.05 * np.sin(2 * np.pi * 0.2 * self.time)
        
        # Add heartbeats
        beat_interval = 60 / 75  # 75 BPM default
        beat_samples = int(beat_interval * self.sampling_rate)
        
        position_in_beat = self.beat_counter % beat_samples
        
        # Generate a beat if at start of beat
        if position_in_beat == 0:
            # Create a synthetic beat
            beat = np.zeros(beat_samples)
            r_peak = int(0.2 * self.sampling_rate)  # R-peak at 200ms
            
            # P wave
            p_start = max(0, r_peak - int(0.15 * self.sampling_rate))
            p_end = r_peak - int(0.05 * self.sampling_rate)
            if p_start < p_end:
                beat[p_start:p_end] += 0.15 * np.hanning(p_end - p_start)
            
            # QRS complex
            qrs_start = max(0, r_peak - int(0.03 * self.sampling_rate))
            qrs_end = min(beat_samples, r_peak + int(0.04 * self.sampling_rate))
            if qrs_start < qrs_end:
                beat[qrs_start:qrs_end] += 1.0 * np.hanning(qrs_end - qrs_start)
            
            # T wave
            t_start = min(beat_samples, r_peak + int(0.15 * self.sampling_rate))
            t_end = min(beat_samples, r_peak + int(0.35 * self.sampling_rate))
            if t_start < t_end:
                beat[t_start:t_end] += 0.25 * np.hanning(t_end - t_start)
            
            self.current_beat = beat
        
        # Get sample from current beat
        if position_in_beat < len(self.current_beat):
            ecg += self.current_beat[position_in_beat]
        
        # Add noise
        ecg += 0.03 * np.random.randn(self.chunk_size)
        
        self.time += duration
        self.beat_counter += self.chunk_size
        
        return ecg.tolist()
    
    def _stream_loop(self):
        """Main streaming loop"""
        start_time = time.time()
        
        while self.running:
            # Check duration limit
            if self.duration > 0 and (time.time() - start_time) > self.duration:
                break
            
            # Generate ECG chunk
            ecg_chunk = self._generate_ecg_chunk()
            
            # Get prediction
            from .routes import model
            prediction = model.predict(np.array(ecg_chunk))
            
            # Emit to client
            self.socketio.emit('ecg_data', {
                'waveform': ecg_chunk,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }, room=self.client_id)
            
            # Simulate real-time delay
            time.sleep(self.chunk_size / self.sampling_rate)
        
        # Stream finished
        self.socketio.emit('stream_ended', {
            'message': 'Stream completed'
        }, room=self.client_id)
        
        self.running = False