/**
 * Plotly ECG Waveform Visualization
 * Real-time ECG plotting with Plotly
 */

class ECGPlotlyVisualizer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            samplingRate: options.samplingRate || 360,
            durationSeconds: options.durationSeconds || 5,
            lineColor: options.lineColor || '#00ff00',
            backgroundColor: options.backgroundColor || '#000',
            ...options
        };
        
        this.waveformData = [];
        this.maxDataPoints = this.options.durationSeconds * this.options.samplingRate;
        this.isAnimating = false;
        this.animationFrame = null;
        
        this.initPlot();
    }
    
    initPlot() {
        const trace = {
            x: [],
            y: [],
            mode: 'lines',
            name: 'ECG',
            line: {
                color: this.options.lineColor,
                width: 2,
                shape: 'spline'
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(0, 255, 0, 0.1)'
        };
        
        const layout = {
            title: {
                text: 'Real-time ECG Waveform',
                font: { color: 'white' }
            },
            xaxis: {
                title: 'Time (seconds)',
                range: [0, this.options.durationSeconds],
                gridcolor: '#333',
                color: 'white'
            },
            yaxis: {
                title: 'Amplitude (mV)',
                range: [-1.5, 1.5],
                gridcolor: '#333',
                color: 'white'
            },
            plot_bgcolor: this.options.backgroundColor,
            paper_bgcolor: this.options.backgroundColor,
            showlegend: false,
            margin: { t: 40, l: 60, r: 30, b: 40 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };
        
        Plotly.newPlot(this.container, [trace], layout, config);
    }
    
    addDataPoint(value, timestamp = null) {
        const time = timestamp || (this.waveformData.length / this.options.samplingRate);
        
        this.waveformData.push({
            x: time,
            y: value
        });
        
        // Keep only recent data
        if (this.waveformData.length > this.maxDataPoints) {
            this.waveformData.shift();
        }
        
        this.updatePlot();
    }
    
    addDataChunk(chunk) {
        for (let i = 0; i < chunk.length; i++) {
            const time = (this.waveformData.length + i) / this.options.samplingRate;
            this.waveformData.push({
                x: time,
                y: chunk[i]
            });
        }
        
        // Keep only recent data
        if (this.waveformData.length > this.maxDataPoints) {
            const excess = this.waveformData.length - this.maxDataPoints;
            this.waveformData = this.waveformData.slice(excess);
        }
        
        this.updatePlot();
    }
    
    updatePlot() {
        const xData = this.waveformData.map(d => d.x);
        const yData = this.waveformData.map(d => d.y);
        
        const update = {
            x: [xData],
            y: [yData]
        };
        
        Plotly.animate(this.container, update, {
            transition: { duration: 0 },
            frame: { duration: 0 }
        });
    }
    
    clear() {
        this.waveformData = [];
        this.updatePlot();
    }
    
    setColor(color) {
        const update = {
            'line.color': color
        };
        Plotly.restyle(this.container, update);
    }
    
    setYRange(min, max) {
        const update = {
            'yaxis.range': [min, max]
        };
        Plotly.relayout(this.container, update);
    }
    
    setDuration(seconds) {
        this.options.durationSeconds = seconds;
        this.maxDataPoints = seconds * this.options.samplingRate;
        
        // Trim data if needed
        if (this.waveformData.length > this.maxDataPoints) {
            this.waveformData = this.waveformData.slice(-this.maxDataPoints);
            this.updatePlot();
        }
        
        // Update x-axis range
        const update = {
            'xaxis.range': [Math.max(0, this.waveformData[this.waveformData.length - 1]?.x - seconds || 0), 
                           this.waveformData[this.waveformData.length - 1]?.x || seconds]
        };
        Plotly.relayout(this.container, update);
    }
    
    exportAsImage() {
        Plotly.downloadImage(this.container, {
            format: 'png',
            width: 1200,
            height: 600,
            filename: 'ecg_waveform'
        });
    }
    
    startAnimation() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        
        let lastTime = 0;
        const animate = (currentTime) => {
            if (!this.isAnimating) return;
            
            const delta = currentTime - lastTime;
            if (delta > 50) { // Update at ~20fps
                this.updatePlot();
                lastTime = currentTime;
            }
            
            this.animationFrame = requestAnimationFrame(animate);
        };
        
        this.animationFrame = requestAnimationFrame(animate);
    }
    
    stopAnimation() {
        this.isAnimating = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    destroy() {
        this.stopAnimation();
        Plotly.purge(this.container);
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ECGPlotlyVisualizer;
}