import { useEffect, useState } from 'react';
import axios from 'axios';

export default function TrainingProgress({ taskId, onComplete, onError }) {
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('Initializing...');
    const [isComplete, setIsComplete] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!taskId) return;

        const pollInterval = setInterval(async () => {
            try {
                const response = await axios.get(`/api/v1/models/task/${taskId}`);
                const data = response.data;

                setProgress(data.progress || 0);
                setStatus(data.message || 'Training in progress...');

                if (data.status === 'SUCCESS') {
                    setIsComplete(true);
                    clearInterval(pollInterval);
                    if (onComplete) onComplete(data.result);
                } else if (data.status === 'FAILURE') {
                    setError(data.message || 'Training failed');
                    clearInterval(pollInterval);
                    if (onError) onError(data.error);
                }
            } catch (err) {
                console.error('Failed to fetch task status:', err);
                setError('Failed to fetch training status');
                clearInterval(pollInterval);
                if (onError) onError(err);
            }
        }, 2000); // Poll every 2 seconds

        return () => clearInterval(pollInterval);
    }, [taskId, onComplete, onError]);

    if (error) {
        return (
            <div className="card" style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                <div style={{ textAlign: 'center', padding: 'var(--spacing-xl)' }}>
                    <div style={{ fontSize: '3rem', marginBottom: 'var(--spacing-md)' }}>❌</div>
                    <h3 style={{ color: 'var(--accent-danger)', marginBottom: 'var(--spacing-sm)' }}>Training Failed</h3>
                    <p className="text-muted">{error}</p>
                </div>
            </div>
        );
    }

    if (isComplete) {
        return (
            <div className="card" style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                <div style={{ textAlign: 'center', padding: 'var(--spacing-xl)' }}>
                    <div style={{ fontSize: '3rem', marginBottom: 'var(--spacing-md)' }}>✅</div>
                    <h3 style={{ color: 'var(--accent-success)', marginBottom: 'var(--spacing-sm)' }}>Training Complete!</h3>
                    <p className="text-muted">Your model has been trained successfully</p>
                </div>
            </div>
        );
    }

    return (
        <div className="card">
            <div style={{ padding: 'var(--spacing-lg)' }}>
                <div style={{ marginBottom: 'var(--spacing-lg)', textAlign: 'center' }}>
                    <div className="spinner" style={{ margin: '0 auto var(--spacing-md)' }}></div>
                    <h3 style={{ marginBottom: 'var(--spacing-sm)' }}>Training in Progress</h3>
                    <p className="text-muted">{status}</p>
                </div>

                {/* Progress Bar */}
                <div style={{ marginBottom: 'var(--spacing-md)' }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: 'var(--spacing-xs)',
                        fontSize: '0.875rem'
                    }}>
                        <span className="text-muted">Progress</span>
                        <span className="font-semibold" style={{ color: 'var(--accent-primary)' }}>
                            {progress}%
                        </span>
                    </div>
                    <div style={{
                        width: '100%',
                        height: '8px',
                        background: 'rgba(255,255,255,0.1)',
                        borderRadius: 'var(--radius-full)',
                        overflow: 'hidden'
                    }}>
                        <div
                            style={{
                                width: `${progress}%`,
                                height: '100%',
                                background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))',
                                transition: 'width 0.5s ease',
                                borderRadius: 'var(--radius-full)'
                            }}
                        />
                    </div>
                </div>

                <div className="card" style={{ background: 'rgba(139, 92, 246, 0.1)', textAlign: 'center' }}>
                    <p className="text-muted" style={{ marginBottom: 0, fontSize: '0.875rem' }}>
                        💡 Training may take several minutes. Feel free to navigate to other pages.
                    </p>
                </div>
            </div>
        </div>
    );
}
