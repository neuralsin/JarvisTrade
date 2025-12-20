import { useState, useEffect } from 'react';
import axios from 'axios';

export default function LiveTrading() {
    const [settings, setSettings] = useState(null);
    const [killSwitch, setKillSwitch] = useState(false);
    const [autoExecute, setAutoExecute] = useState(false);
    const [password, setPassword] = useState('');
    const [showPasswordModal, setShowPasswordModal] = useState(false);
    const [pendingAction, setPendingAction] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        try {
            const [killResponse, userResponse] = await Promise.all([
                axios.get('/api/v1/admin/kill-switch'),
                axios.get('/api/v1/auth/me')
            ]);

            setKillSwitch(killResponse.data.kill_switch_enabled);
            setAutoExecute(userResponse.data.auto_execute);
        } catch (error) {
            console.error('Failed to fetch settings:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleToggleKillSwitch = () => {
        setPendingAction('killswitch');
        setShowPasswordModal(true);
    };

    const handleToggleAutoExecute = () => {
        setPendingAction('autoexecute');
        setShowPasswordModal(true);
    };

    const confirmAction = async () => {
        try {
            if (pendingAction === 'killswitch') {
                await axios.post('/api/v1/admin/kill-switch', {
                    enabled: !killSwitch,
                    password
                });
                setKillSwitch(!killSwitch);
            } else if (pendingAction === 'autoexecute') {
                await axios.post(`/api/v1/admin/auto-execute?enabled=${!autoExecute}&password=${password}`);
                setAutoExecute(!autoExecute);
            }

            setShowPasswordModal(false);
            setPassword('');
            setPendingAction(null);
        } catch (error) {
            alert(error.response?.data?.detail || 'Action failed');
        }
    };

    if (loading) {
        return <div className="flex justify-center items-center" style={{ height: '400px' }}><div className="spinner"></div></div>;
    }

    return (
        <div className="fade-in">
            <h1 className="mb-lg">Live Trading</h1>

            {/* Warning Banner */}
            <div className="card" style={{ background: 'rgba(239, 68, 68, 0.1)', borderColor: 'var(--accent-danger)', marginBottom: 'var(--spacing-xl)' }}>
                <div className="flex items-center gap-md">
                    <span style={{ fontSize: '2rem' }}>⚠️</span>
                    <div>
                        <h3 className="text-danger" style={{ marginBottom: 'var(--spacing-sm)' }}>Live Trading Warning</h3>
                        <p className="text-sm">Live trading executes real trades with real money. Ensure you understand the risks before enabling.
                        </p>
                    </div>
                </div>
            </div>

            {/* Status Cards */}
            <div className="grid grid-cols-2 mb-lg">
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Kill Switch</h3>
                        <span className={`badge ${killSwitch ? 'badge-danger' : 'badge-success'}`}>
                            {killSwitch ? 'ENABLED' : 'DISABLED'}
                        </span>
                    </div>
                    <p className="text-muted mb-md">
                        Emergency stop for all trading activity. When enabled, no trades will be executed.
                    </p>
                    <button
                        className={`btn ${killSwitch ? 'btn-success' : 'btn-danger'}`}
                        onClick={handleToggleKillSwitch}
                        style={{ width: '100%' }}
                    >
                        {killSwitch ? 'Disable Kill Switch' : 'Enable Kill Switch'}
                    </button>
                </div>

                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Auto Execute</h3>
                        <span className={`badge ${autoExecute ? 'badge-success' : 'badge-warning'}`}>
                            {autoExecute ? 'ON' : 'OFF'}
                        </span>
                    </div>
                    <p className="text-muted mb-md">
                        Allow the system to automatically execute trades based on model signals.
                    </p>
                    <button
                        className={`btn ${autoExecute ? 'btn-outline' : 'btn-success'}`}
                        onClick={handleToggleAutoExecute}
                        disabled={killSwitch}
                        style={{ width: '100%' }}
                    >
                        {autoExecute ? 'Disable Auto Execute' : 'Enable Auto Execute'}
                    </button>
                </div>
            </div>

            {/* Kite Integration */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Zerodha Kite Integration</h3>
                </div>
                <p className="text-muted mb-md">
                    Connect your Zerodha Kite account to enable live trading.
                </p>
                <button className="btn btn-primary">
                    Configure Kite Credentials
                </button>
            </div>

            {/* Password Modal */}
            {showPasswordModal && (
                <div className="modal-overlay" onClick={() => setShowPasswordModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <h3 className="mb-lg">Confirm Action</h3>
                        <p className="text-muted mb-md">Enter your password to confirm this action</p>

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
                                    setPendingAction(null);
                                }}
                            >
                                Cancel
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={confirmAction}
                                disabled={!password}
                            >
                                Confirm
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
