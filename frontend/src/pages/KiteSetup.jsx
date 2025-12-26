import { useState, useEffect } from 'react';
import axios from 'axios';

export default function KiteSetup() {
    const [status, setStatus] = useState('checking');
    const [message, setMessage] = useState('');
    const [apiKey, setApiKey] = useState('');
    const [apiSecret, setApiSecret] = useState('');

    useEffect(() => {
        checkStatus();
    }, []);

    const checkStatus = async () => {
        try {
            // Check if credentials are already saved
            const response = await axios.get('/api/v1/auth/kite/login-url');
            setStatus('ready');
            setMessage('Kite API credentials are configured! Click below to authorize.');
        } catch (error) {
            if (error.response?.status === 400) {
                setStatus('needs_credentials');
                setMessage('Kite API credentials not found. Enter them below or add to .env file.');
            } else {
                setStatus('error');
                setMessage('Error checking status: ' + error.message);
            }
        }
    };

    const handleSaveCredentials = async () => {
        if (!apiKey || !apiSecret) {
            alert('Please enter both API Key and API Secret');
            return;
        }

        try {
            await axios.post('/api/v1/auth/kite/credentials', {
                api_key: apiKey,
                api_secret: apiSecret
            });
            setStatus('ready');
            setMessage('✅ Credentials saved! Now click "Authorize with Kite" below.');
        } catch (error) {
            alert('Failed to save credentials: ' + error.message);
        }
    };

    const handleAuthorize = async () => {
        try {
            const response = await axios.get('/api/v1/auth/kite/login-url');
            const loginUrl = response.data.login_url;

            // Open Kite login in same window
            window.location.href = loginUrl;
        } catch (error) {
            alert('Failed to get authorization URL: ' + error.message);
        }
    };

    return (
        <div className="fade-in" style={{ maxWidth: '600px', margin: '0 auto', padding: 'var(--spacing-2xl)' }}>
            <div className="card">
                <h1 style={{ marginBottom: 'var(--spacing-lg)', textAlign: 'center' }}>
                    🔐 Kite API Setup
                </h1>

                {/* Status Message */}
                <div
                    className="card"
                    style={{
                        background: status === 'ready' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(59, 130, 246, 0.1)',
                        border: `1px solid ${status === 'ready' ? '#10b981' : '#3b82f6'}`,
                        marginBottom: 'var(--spacing-lg)',
                        textAlign: 'center'
                    }}
                >
                    <p>{message}</p>
                </div>

                {/* Credentials Input (if needed) */}
                {status === 'needs_credentials' && (
                    <div style={{ marginBottom: 'var(--spacing-lg)' }}>
                        <h3 style={{ marginBottom: 'var(--spacing-md)' }}>Enter Kite API Credentials</h3>
                        <p className="text-sm text-muted" style={{ marginBottom: 'var(--spacing-md)' }}>
                            Get these from <a href="https://developers.kite.trade/" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-primary)' }}>kite.trade</a>
                        </p>

                        <div className="form-group">
                            <label className="form-label">API Key</label>
                            <input
                                type="text"
                                className="form-input"
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                placeholder="e.g., abc123xyz456"
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">API Secret</label>
                            <input
                                type="password"
                                className="form-input"
                                value={apiSecret}
                                onChange={(e) => setApiSecret(e.target.value)}
                                placeholder="e.g., def789ghi012"
                            />
                        </div>

                        <button
                            className="btn btn-primary"
                            onClick={handleSaveCredentials}
                            style={{ width: '100%' }}
                        >
                            Save Credentials
                        </button>
                    </div>
                )}

                {/* Authorization Button */}
                {status === 'ready' && (
                    <div>
                        <button
                            className="btn btn-primary"
                            onClick={handleAuthorize}
                            style={{
                                width: '100%',
                                fontSize: '1.1rem',
                                padding: 'var(--spacing-md) var(--spacing-lg)'
                            }}
                        >
                            🚀 Authorize with Kite
                        </button>

                        <div className="text-sm text-muted" style={{ marginTop: 'var(--spacing-lg)', textAlign: 'center' }}>
                            <p>You'll be redirected to Zerodha Kite to login and authorize.</p>
                            <p>After authorization, you can train models on ANY Indian stock!</p>
                        </div>
                    </div>
                )}

                {/* Instructions */}
                <div className="card" style={{ background: 'var(--bg-tertiary)', marginTop: 'var(--spacing-xl)' }}>
                    <h4 style={{ marginBottom: 'var(--spacing-sm)' }}>📋 What This Does</h4>
                    <ul style={{ marginLeft: 'var(--spacing-lg)', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                        <li>Connects JarvisTrade to Zerodha Kite API</li>
                        <li>Enables training on stocks like TATAELXSI, HAL, etc.</li>
                        <li>Provides access to entire NSE/BSE stock universe</li>
                        <li>Bypasses Yahoo Finance limitations</li>
                    </ul>
                </div>
            </div>
        </div>
    );
}
