"""
Specialized ECG visualization with PQRST annotation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, List, Tuple, Dict
from scipy.signal import find_peaks


class PQRSTAnnotator:
    """
    Annotate PQRST points on ECG signal
    
    Identifies and visualizes:
    - P wave (atrial depolarization)
    - QRS complex (ventricular depolarization)
    - T wave (ventricular repolarization)
    - ST segment
    - PR interval
    - QT interval
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize PQRST annotator
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def detect_pqrst(self, beat: np.ndarray, r_peak_idx: int) -> Dict:
        """
        Detect P, Q, R, S, T points in a beat
        
        Args:
            beat: Single ECG beat
            r_peak_idx: Index of R-peak within beat
            
        Returns:
            Dictionary with detected points
        """
        points = {
            'p_idx': None,
            'q_idx': None,
            'r_idx': r_peak_idx,
            's_idx': None,
            't_idx': None,
            'p_onset': None,
            'p_offset': None,
            'qrs_onset': None,
            'qrs_offset': None,
            't_onset': None,
            't_offset': None
        }
        
        # Search windows (in samples)
        # Q wave: 20-60 ms before R
        q_start = max(0, r_peak_idx - int(0.06 * self.sampling_rate))
        q_end = max(0, r_peak_idx - int(0.01 * self.sampling_rate))
        if q_start < q_end:
            q_region = beat[q_start:q_end]
            if len(q_region) > 0:
                q_idx_rel = np.argmin(q_region)
                points['q_idx'] = q_start + q_idx_rel
                
        # S wave: 20-60 ms after R
        s_start = min(len(beat) - 1, r_peak_idx + int(0.01 * self.sampling_rate))
        s_end = min(len(beat) - 1, r_peak_idx + int(0.06 * self.sampling_rate))
        if s_start < s_end:
            s_region = beat[s_start:s_end]
            if len(s_region) > 0:
                s_idx_rel = np.argmin(s_region)
                points['s_idx'] = s_start + s_idx_rel
                
        # P wave: 200-400 ms before R
        p_start = max(0, r_peak_idx - int(0.45 * self.sampling_rate))
        p_end = max(0, r_peak_idx - int(0.15 * self.sampling_rate))
        if p_start < p_end:
            p_region = beat[p_start:p_end]
            if len(p_region) > 0:
                p_idx_rel = np.argmax(p_region)
                points['p_idx'] = p_start + p_idx_rel
                
        # T wave: 150-400 ms after R
        t_start = min(len(beat) - 1, r_peak_idx + int(0.15 * self.sampling_rate))
        t_end = min(len(beat) - 1, r_peak_idx + int(0.45 * self.sampling_rate))
        if t_start < t_end:
            t_region = beat[t_start:t_end]
            if len(t_region) > 0:
                t_idx_rel = np.argmax(t_region)
                points['t_idx'] = t_start + t_idx_rel
                
        # Detect onsets and offsets
        if points['p_idx']:
            points['p_onset'] = self._find_onset(beat, points['p_idx'], direction='left')
            points['p_offset'] = self._find_offset(beat, points['p_idx'], direction='right')
            
        if points['q_idx'] and points['s_idx']:
            points['qrs_onset'] = points['q_idx']
            points['qrs_offset'] = points['s_idx']
            
        if points['t_idx']:
            points['t_onset'] = self._find_onset(beat, points['t_idx'], direction='left')
            points['t_offset'] = self._find_offset(beat, points['t_idx'], direction='right')
            
        return points
    
    def _find_onset(self, signal: np.ndarray, peak_idx: int, 
                   direction: str = 'left') -> int:
        """Find onset of a wave (where signal returns to baseline)"""
        baseline = np.median(signal[:50])  # Estimate baseline from start
        
        if direction == 'left':
            search_range = range(peak_idx, max(0, peak_idx - int(0.1 * self.sampling_rate)), -1)
        else:
            search_range = range(peak_idx, min(len(signal), peak_idx + int(0.1 * self.sampling_rate)))
            
        for i in search_range:
            if abs(signal[i] - baseline) < 0.05 * abs(signal[peak_idx] - baseline):
                return i
        return peak_idx
    
    def _find_offset(self, signal: np.ndarray, peak_idx: int,
                    direction: str = 'right') -> int:
        """Find offset of a wave"""
        return self._find_onset(signal, peak_idx, direction)
    
    def annotate_beat(self, beat: np.ndarray, r_peak_idx: int,
                     ax: plt.Axes, color: str = 'blue') -> Dict:
        """
        Annotate a single beat with PQRST markers
        
        Args:
            beat: ECG beat signal
            r_peak_idx: R-peak index within beat
            ax: Matplotlib axes
            color: Marker color
            
        Returns:
            Dictionary with annotation positions
        """
        points = self.detect_pqrst(beat, r_peak_idx)
        
        time = np.arange(len(beat)) / self.sampling_rate * 1000  # Convert to ms
        
        # Plot signal
        ax.plot(time, beat, 'k-', linewidth=1.5)
        
        # Annotate PQRST points
        annotations = []
        
        if points['p_idx']:
            ax.plot(time[points['p_idx']], beat[points['p_idx']], 
                   'go', markersize=8, label='P wave')
            annotations.append(('P', points['p_idx']))
            
        if points['q_idx']:
            ax.plot(time[points['q_idx']], beat[points['q_idx']], 
                   'yo', markersize=8, label='Q wave')
            annotations.append(('Q', points['q_idx']))
            
        # R peak
        ax.plot(time[points['r_idx']], beat[points['r_idx']], 
               'ro', markersize=10, label='R peak')
        annotations.append(('R', points['r_idx']))
        
        if points['s_idx']:
            ax.plot(time[points['s_idx']], beat[points['s_idx']], 
                   'mo', markersize=8, label='S wave')
            annotations.append(('S', points['s_idx']))
            
        if points['t_idx']:
            ax.plot(time[points['t_idx']], beat[points['t_idx']], 
                   'co', markersize=8, label='T wave')
            annotations.append(('T', points['t_idx']))
            
        # Highlight intervals
        if points['qrs_onset'] and points['qrs_offset']:
            ax.axvspan(time[points['qrs_onset']], time[points['qrs_offset']],
                      alpha=0.2, color='red', label='QRS complex')
            
        if points['p_onset'] and points['p_offset']:
            ax.axvspan(time[points['p_onset']], time[points['p_offset']],
                      alpha=0.2, color='green', label='P wave')
            
        if points['t_onset'] and points['t_offset']:
            ax.axvspan(time[points['t_onset']], time[points['t_offset']],
                      alpha=0.2, color='cyan', label='T wave')
            
        # Add text labels
        for label, idx in annotations:
            ax.annotate(label, 
                       (time[idx], beat[idx]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
            
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title('ECG Beat with PQRST Annotation')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        return points


class ECGVisualizer:
    """
    Comprehensive ECG visualization tool
    
    Features:
    - Multi-lead ECG display
    - PQRST annotation
    - Comparison plots
    - Spectrogram generation
    - Rhythm strip visualization
    """
    
    def __init__(self, sampling_rate: int = 360, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize ECG visualizer
        
        Args:
            sampling_rate: ECG sampling rate in Hz
            figsize: Default figure size
        """
        self.sampling_rate = sampling_rate
        self.figsize = figsize
        self.annotator = PQRSTAnnotator(sampling_rate)
        
    def plot_multi_lead(self, signals: np.ndarray,
                       lead_names: List[str],
                       duration_seconds: float = 5,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple ECG leads
        
        Args:
            signals: Array of signals (n_leads, n_samples)
            lead_names: Names of leads
            duration_seconds: Duration to display
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_leads = signals.shape[0]
        samples_to_plot = int(duration_seconds * self.sampling_rate)
        
        fig, axes = plt.subplots(n_leads, 1, figsize=self.figsize, sharex=True)
        
        if n_leads == 1:
            axes = [axes]
            
        time = np.arange(samples_to_plot) / self.sampling_rate
        
        for i, (ax, signal) in enumerate(zip(axes, signals)):
            ax.plot(time, signal[:samples_to_plot], 'b-', linewidth=1)
            ax.set_ylabel(f'{lead_names[i]} (mV)')
            ax.set_title(f'Lead {lead_names[i]}')
            ax.grid(True, alpha=0.3)
            
            # Add scale bars
            ax.annotate('1 mV', xy=(0.02, 0.9), xycoords='axes fraction',
                       bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
        axes[-1].set_xlabel('Time (seconds)')
        fig.suptitle('Multi-lead ECG Recording', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_rhythm_strip(self, signal: np.ndarray,
                         r_peaks: np.ndarray,
                         duration_seconds: float = 10,
                         beat_window_ms: int = 500,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot rhythm strip with overlaid beats
        
        Args:
            signal: ECG signal
            r_peaks: R-peak indices
            duration_seconds: Duration to display
            beat_window_ms: Window around each beat (ms)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        samples_to_plot = int(duration_seconds * self.sampling_rate)
        window_samples = int(beat_window_ms * self.sampling_rate / 1000)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Full rhythm strip
        time = np.arange(samples_to_plot) / self.sampling_rate
        ax1.plot(time, signal[:samples_to_plot], 'b-', linewidth=1)
        ax1.scatter(r_peaks[r_peaks < samples_to_plot] / self.sampling_rate,
                   signal[r_peaks[r_peaks < samples_to_plot]],
                   c='red', s=30, zorder=5, marker='^')
        ax1.set_ylabel('Amplitude (mV)')
        ax1.set_title('Rhythm Strip with R-peaks')
        ax1.grid(True, alpha=0.3)
        
        # Overlaid beats        beats = []
        for peak in r_peaks:
            if peak - window_samples > 0 and peak + window_samples < len(signal):
                beat = signal[peak - window_samples:peak + window_samples]
                beats.append(beat)
                
        if beats:
            beats = np.array(beats)
            beat_time = np.arange(-window_samples, window_samples) / self.sampling_rate * 1000
            
            for beat in beats[:20]:  # Plot first 20 beats
                ax2.plot(beat_time, beat, 'b-', alpha=0.5, linewidth=0.8)
                
            # Plot mean beat
            mean_beat = np.mean(beats, axis=0)
            std_beat = np.std(beats, axis=0)
            ax2.plot(beat_time, mean_beat, 'r-', linewidth=2, label='Mean Beat')
            ax2.fill_between(beat_time, mean_beat - std_beat, mean_beat + std_beat,
                            alpha=0.3, color='red', label='±1 STD')
            
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Amplitude (mV)')
            ax2.set_title('Overlaid Beats (aligned at R-peak)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        ax1.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_spectrogram(self, signal: np.ndarray,
                        duration_seconds: float = 10,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ECG spectrogram for time-frequency analysis
        
        Args:
            signal: ECG signal
            duration_seconds: Duration to display
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        samples_to_plot = int(duration_seconds * self.sampling_rate)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Time domain
        time = np.arange(samples_to_plot) / self.sampling_rate
        ax1.plot(time, signal[:samples_to_plot], 'b-', linewidth=1)
        ax1.set_ylabel('Amplitude (mV)')
        ax1.set_title('ECG Signal (Time Domain)')
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(signal[:samples_to_plot], fs=self.sampling_rate,
                               nperseg=256, noverlap=128)
        
        im = ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                           shading='gouraud', cmap='viridis')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Spectrogram (Time-Frequency Representation)')
        ax2.set_ylim(0, 50)  # ECG frequencies up to 50 Hz
        
        plt.colorbar(im, ax=ax2, label='Power/Frequency (dB/Hz)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_annotated_beat(self, beat: np.ndarray, r_peak_idx: int,
                           title: str = "Annotated ECG Beat",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a single beat with PQRST annotation
        
        Args:
            beat: ECG beat signal
            r_peak_idx: R-peak index within beat
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        points = self.annotator.annotate_beat(beat, r_peak_idx, ax)
        ax.set_title(title)
        
        # Add interval text
        if points['qrs_onset'] and points['qrs_offset']:
            qrs_duration = (points['qrs_offset'] - points['qrs_onset']) / self.sampling_rate * 1000
            ax.text(0.02, 0.95, f'QRS Duration: {qrs_duration:.1f} ms',
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
            
        if points['q_idx'] and points['t_idx']:
            qt_interval = (points['t_idx'] - points['q_idx']) / self.sampling_rate * 1000
            ax.text(0.02, 0.88, f'QT Interval: {qt_interval:.1f} ms',
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


if __name__ == "__main__":
    # Test ECG visualizer
    # Generate synthetic ECG beat
    t = np.linspace(-0.2, 0.5, int(0.7 * 360))
    beat = np.zeros_like(t)
    
    # P wave
    p_idx = np.argmin(np.abs(t - (-0.15)))
    beat[p_idx:p_idx+15] = 0.15 * np.hanning(15)
    
    # QRS complex
    r_idx = np.argmin(np.abs(t - 0))
    beat[r_idx-5:r_idx+15] = 1.0 * np.hanning(20)
    
    # S wave
    s_idx = np.argmin(np.abs(t - 0.04))
    beat[s_idx:s_idx+10] = -0.3 * np.hanning(10)
    
    # T wave
    t_idx = np.argmin(np.abs(t - 0.3))
    beat[t_idx:t_idx+25] = 0.25 * np.hanning(25)
    
    # Add noise
    beat += 0.02 * np.random.randn(len(beat))
    
    # Create visualizer
    visualizer = ECGVisualizer(sampling_rate=360)
    
    # Plot annotated beat
    visualizer.plot_annotated_beat(beat, r_idx)
    plt.show()
    
    print("ECG visualizer test completed")