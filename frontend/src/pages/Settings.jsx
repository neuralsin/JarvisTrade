import { useState, useEffect } from 'react';
import axios from 'axios';

export default function Settings() {
    const [mode, setMode] = useState('paper');
    const [parameters, setParameters] = useState(null);
    const [editedParams, setEditedParams] = useState(null);
    const [password, setPassword] = useState('');
    const [showPasswordModal, setShowPasswordModal] = useState(false);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        fetchParameters();
    }, [mode]);

    const fetchParameters = async () => {
        try {
            const response = await axios.get(`/api/v1/settings/parameters?mode=${mode}`);
            setParameters(response.data.parameters);
            setEditedParams(response.data.parameters);
        } catch (error) {
            console.error('Failed to fetch parameters:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!password) {
            alert('Please enter your password');
            return;
        }

        setSaving(true);
        try {
            await axios.post('/api/v1/settings/parameters', {
                parameters: editedParams,
                password,
                mode
            });

            setParameters(editedParams);
            setShowPasswordModal(false);
            setPassword('');
            alert('Parameters updated successfully!');
        } catch (error) {
            alert(error.response?.data?.detail || 'Failed to update parameters');
        } finally {
            setSaving(false);
        }
    };

    const handleReset = async () => {
        if (!password) {
            alert('Please enter your password');
            return;
        }

        try {
            await axios.post(`/api/v1/settings/parameters/reset?mode=${mode}&password=${password}`);
            await fetchParameters();
            setShowPasswordModal(false);
            setPassword('');
            alert('Parameters reset to defaults!');
        } catch (error) {
            alert(error.response?.data?.detail || 'Failed to reset parameters');
        }
    };

    const hasChanges = JSON.stringify(parameters) !== JSON.stringify(editedParams);

    if (loading) {
        return <div className="flex justify-center items-center" style={{ height: '400px' }}><div className="spinner"></div></div>;
    }

    return (
        <div className="fade-in">
            <h1 className="mb-lg">Trading Parameters</h1>

            {/* Mode Selector */}
            <div className="card mb-lg">
                <div className="flex gap-md">
                    <button
                        className={`btn ${mode === 'paper' ? 'btn-primary' : 'btn-outline'}`}
                        onClick={() => setMode('paper')}
                    >
                        📝 Paper Trading
                    </button>
                    <button
                        className={`btn ${mode === 'live' ? 'btn-primary' : 'btn-outline'}`}
                        onClick={() => setMode('live')}
                    >
                        ⚡ Live Trading
                    </button>
                </div>
                <p className="text-sm text-muted mt-md">
                    Configure parameters separately for paper and live trading modes
                </p>
            </div>

            {/* Parameters Form */}
            <div className="grid grid-cols-2">
                <div className="card">
                    <h3 className="card-title">Capital & Risk</h3>

                    <div className="form-group">
                        <label className="form-label">Account Capital (₹)</label>
                        <input
                            type="number"
                            className="form-input"
                            value={editedParams?.account_capital || 0}
                            onChange={(e) => setEditedParams({ ...editedParams, account_capital: parseFloat(e.target.value) })}
                            min="1000"
                            step="1000"
                        />
                        <p className="text-xs text-muted mt-sm">Starting capital for trading</p>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Risk Per Trade (%)</label>
                        <input
                            type="number"
                            className="form-input"
                            value={(editedParams?.risk_per_trade || 0) * 100}
                            onChange={(e) => setEditedParams({ ...editedParams, risk_per_trade: parseFloat(e.target.value) / 100 })}
                            min="0.1"
                            max="10"
                            step="0.1"
                        />
                        <p className="text-xs text-muted mt-sm">
                            Current: {((editedParams?.risk_per_trade || 0) * 100).toFixed(1)}% = ₹{((editedParams?.account_capital || 0) * (editedParams?.risk_per_trade || 0)).toFixed(0)} per trade
                        </p>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Max Daily Loss (%)</label>
                        <input
                            type="number"
                            className="form-input"
                            value={(editedParams?.max_daily_loss || 0) * 100}
                            onChange={(e) => setEditedParams({ ...editedParams, max_daily_loss: parseFloat(e.target.value) / 100 })}
                            min="0.5"
                            max="20"
                            step="0.5"
                        />
                        <p className="text-xs text-muted mt-sm">
                            Trading stops if daily loss exceeds {((editedParams?.max_daily_loss || 0) * 100).toFixed(1)}%
                        </p>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Max Trades Per Day</label>
                        <input
                            type="number"
                            className="form-input"
                            value={editedParams?.max_trades_per_day || 0}
                            onChange={(e) => setEditedParams({ ...editedParams, max_trades_per_day: parseInt(e.target.value) })}
                            min="1"
                            max="50"
                            step="1"
                        />
                        <p className="text-xs text-muted mt-sm">Maximum number of trades allowed per day</p>
                    </div>
                </div>

                <div className="card">
                    <h3 className="card-title">Entry & Exit Strategy</h3>

                    <div className="form-group">
                        <label className="form-label">Stop Loss Multiplier</label>
                        <input
                            type="number"
                            className="form-input"
                            value={editedParams?.stop_multiplier || 0}
                            onChange={(e) => setEditedParams({ ...editedParams, stop_multiplier: parseFloat(e.target.value) })}
                            min="0.5"
                            max="5"
                            step="0.1"
                        />
                        <p className="text-xs text-muted mt-sm">
                            Stop loss = Entry Price - (ATR × {editedParams?.stop_multiplier?.toFixed(1)})
                        </p>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Target Multiplier</label>
                        <input
                            type="number"
                            className="form-input"
                            value={editedParams?.target_multiplier || 0}
                            onChange={(e) => setEditedParams({ ...editedParams, target_multiplier: parseFloat(e.target.value) })}
                            min="1"
                            max="10"
                            step="0.1"
                        />
                        <p className="text-xs text-muted mt-sm">
                            Target = Entry Price + (Risk Distance × {editedParams?.target_multiplier?.toFixed(1)})
                        </p>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Minimum Probability (%)</label>
                        <input
                            type="number"
                            className="form-input"
                            value={(editedParams?.prob_min || 0) * 100}
                            onChange={(e) => setEditedParams({ ...editedParams, prob_min: parseFloat(e.target.value) / 100 })}
                            min="50"
                            max="95"
                            step="1"
                        />
                        <p className="text-xs text-muted mt-sm">
                            Only trade if model probability ≥ {((editedParams?.prob_min || 0) * 100).toFixed(0)}%
                        </p>
                    </div>

                    <div className="form-group">
                        <label className="form-label">Strong Signal Threshold (%)</label>
                        <input
                            type="number"
                            className="form-input"
                            value={(editedParams?.prob_strong || 0) * 100}
                            onChange={(e) => setEditedParams({ ...editedParams, prob_strong: parseFloat(e.target.value) / 100 })}
                            min="60"
                            max="99"
                            step="1"
                        />
                        <p className="text-xs text-muted mt-sm">
                            Signals above {((editedParams?.prob_strong || 0) * 100).toFixed(0)}% are considered strong
                        </p>
                    </div>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="card mt-lg">
                <div className="flex justify-between items-center">
                    <button
                        className="btn btn-outline"
                        onClick={() => {
                            setEditedParams(parameters);
                        }}
                        disabled={!hasChanges}
                    >
                        ↺ Reset Changes
                    </button>

                    <div className="flex gap-md">
                        <button
                            className="btn btn-danger"
                            onClick={() => {
                                if (confirm('Reset all parameters to system defaults?')) {
                                    setShowPasswordModal(true);
                                }
                            }}
                        >
                            Reset to Defaults
                        </button>

                        <button
                            className="btn btn-success"
                            onClick={() => setShowPasswordModal(true)}
                            disabled={!hasChanges}
                        >
                            💾 Save Changes
                        </button>
                    </div>
                </div>

                {hasChanges && (
                    <div className="mt-md" style={{ padding: 'var(--spacing-md)', background: 'rgba(245, 158, 11, 0.1)', borderRadius: 'var(--radius-md)', borderLeft: '3px solid var(--accent-warning)' }}>
                        <p className="text-warning text-sm font-semibold">⚠️ You have unsaved changes</p>
                        <p className="text-xs text-muted">Click "Save Changes" and confirm with your password to apply these parameters</p>
                    </div>
                )}
            </div>

            {/* Password Confirmation Modal */}
            {showPasswordModal && (
                <div className="modal-overlay" onClick={() => setShowPasswordModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <h3 className="mb-lg">Confirm Changes</h3>
                        <p className="text-muted mb-md">
                            Enter your password to confirm parameter changes for <strong>{mode}</strong> trading
                        </p>

                        <div className="form-group">
                            <label className="form-label">Password</label>
                            <input
                                type="password"
                                className="form-input"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="••••••••"
                                autoFocus
                            />
                        </div>

                        <div className="flex gap-md" style={{ justifyContent: 'flex-end' }}>
                            <button
                                className="btn btn-outline"
                                onClick={() => {
                                    setShowPasswordModal(false);
                                    setPassword('');
                                }}
                            >
                                Cancel
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={handleSave}
                                disabled={!password || saving}
                            >
                                {saving ? <div className="spinner" style={{ width: '1rem', height: '1rem' }}></div> : 'Confirm & Save'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
