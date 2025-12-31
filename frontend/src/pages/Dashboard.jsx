import { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

export default function Dashboard() {
    const [mode, setMode] = useState('paper');
    const [data, setData] = useState(null);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [kiteStatus, setKiteStatus] = useState({ hasCredentials: false, hasToken: false });

    useEffect(() => {
        fetchDashboard();
        fetchStats();
        checkKiteStatus();
    }, [mode]);

    const checkKiteStatus = async () => {
        try {
            const response = await axios.get('/api/v1/auth/me');
            console.log('Kite status from API:', response.data);
            setKiteStatus({
                hasCredentials: response.data.has_kite_credentials,
                hasToken: response.data.has_kite_access_token || false,
                source: response.data.kite_credentials_source
            });
        } catch (error) {
            console.log('Auth check failed (likely not logged in), assuming credentials exist from .env');
            // If auth fails, assume credentials are in .env since backend is configured
            setKiteStatus({
                hasCredentials: true, // Assume true if .env is configured
                hasToken: false,
                source: 'env'
            });
        }
    };

    const handleKiteAuthorization = async () => {
        try {
            // Get login URL from backend
            const response = await axios.get('/api/v1/auth/kite/login-url');
            const loginUrl = response.data.login_url;

            // Open in new window
            const authWindow = window.open(
                loginUrl,
                'Kite Authorization',
                'width=800,height=600,left=200,top=100'
            );

            // Poll for completion (optional)
            const pollTimer = setInterval(() => {
                if (authWindow.closed) {
                    clearInterval(pollTimer);
                    checkKiteStatus(); // Refresh status
                    alert('✅ Kite authorization complete! You can now train on any Indian stock.');
                }
            }, 1000);
        } catch (error) {
            if (error.response?.status === 400) {
                alert('❌ Please add your Kite API credentials to the .env file first.\n\nSee kite_api_setup.md for instructions.');
            } else {
                console.error('Failed to get Kite login URL:', error);
                alert('Failed to start Kite authorization. Check console for details.');
            }
        }
    };


    const fetchDashboard = async () => {
        try {
            const response = await axios.get(`/api/v1/dashboard?mode=${mode}`);
            setData(response.data);
        } catch (error) {
            console.error('Failed to fetch dashboard:', error);
        } finally {
            setLoading(false);
        }
    };

    const fetchStats = async () => {
        try {
            const response = await axios.get(`/api/v1/dashboard/stats?mode=${mode}`);
            setStats(response.data);
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    };

    if (loading) {
        return <div className="flex justify-center items-center" style={{ height: '400px' }}><div className="spinner"></div></div>;
    }

    return (
        <div className="fade-in">
            <div className="flex justify-between items-center mb-lg">
                <h1>Dashboard</h1>
                <div className="flex gap-md">
                    {/* Kite Authorization Button */}
                    <button
                        className="btn btn-outline"
                        onClick={handleKiteAuthorization}
                        disabled={kiteStatus.hasToken}
                        style={{
                            borderColor: kiteStatus.hasToken ? '#10b981' : (kiteStatus.hasCredentials ? '#f59e0b' : '#6b7280'),
                            color: kiteStatus.hasToken ? '#10b981' : (kiteStatus.hasCredentials ? '#f59e0b' : '#9ca3af'),
                            cursor: kiteStatus.hasToken ? 'default' : 'pointer'
                        }}
                        title={
                            kiteStatus.hasToken
                                ? 'Kite API connected!'
                                : (kiteStatus.hasCredentials
                                    ? 'Click to authorize Kite API'
                                    : 'Add Kite credentials to .env first')
                        }
                    >
                        {kiteStatus.hasToken
                            ? '✓ Kite Connected'
                            : (kiteStatus.hasCredentials
                                ? '🔑 Authorize Kite'
                                : '○ Connect Kite API')}
                    </button>

                    <button
                        className={`btn ${mode === 'paper' ? 'btn-primary' : 'btn-outline'}`}
                        onClick={() => setMode('paper')}
                    >
                        Paper Trading
                    </button>
                    <button
                        className={`btn ${mode === 'live' ? 'btn-primary' : 'btn-outline'}`}
                        onClick={() => setMode('live')}
                    >
                        Live Trading
                    </button>
                </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-5 mb-lg">
                <div className="card stat-card">
                    <div className="stat-value">{stats?.total_trades || 0}</div>
                    <div className="stat-label">Total Trades</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value" style={{ color: (stats?.total_pnl || 0) >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)' }}>
                        ₹{(stats?.total_pnl || 0).toLocaleString()}
                    </div>
                    <div className="stat-label">Total P&L</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value">{((stats?.win_rate || 0) * 100).toFixed(1)}%</div>
                    <div className="stat-label">Win Rate</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value">₹{(stats?.avg_pnl || 0).toFixed(0)}</div>
                    <div className="stat-label">Avg P&L</div>
                </div>
                {/* V2 Regime Status */}
                <div className="card stat-card" style={{
                    background: stats?.engine_version === 'v2'
                        ? stats?.current_regime === 'TREND_STABLE' ? 'rgba(16, 185, 129, 0.1)'
                            : stats?.current_regime === 'TREND_VOLATILE' ? 'rgba(245, 158, 11, 0.1)'
                                : stats?.current_regime === 'RANGE_QUIET' ? 'rgba(59, 130, 246, 0.1)'
                                    : stats?.current_regime === 'CHOP_PANIC' ? 'rgba(239, 68, 68, 0.1)'
                                        : 'var(--bg-secondary)'
                        : 'var(--bg-secondary)',
                    border: stats?.engine_version === 'v2' ? '1px solid rgba(139, 92, 246, 0.3)' : 'none'
                }}>
                    <div className="stat-value" style={{ fontSize: '1.2rem' }}>
                        {stats?.engine_version === 'v2' ? (
                            <>
                                {stats?.current_regime === 'TREND_STABLE' && '📈 TREND'}
                                {stats?.current_regime === 'TREND_VOLATILE' && '⚡ VOLATILE'}
                                {stats?.current_regime === 'RANGE_QUIET' && '➖ RANGE'}
                                {stats?.current_regime === 'CHOP_PANIC' && '🛑 PANIC'}
                                {!stats?.current_regime && '🎯 V2'}
                            </>
                        ) : (
                            'V1'
                        )}
                    </div>
                    <div className="stat-label">
                        {stats?.engine_version === 'v2' ? 'Market Regime' : 'Engine'}
                    </div>
                </div>
            </div>

            {/* Equity Curve */}
            <div className="card mb-lg">
                <div className="card-header">
                    <h3 className="card-title">Equity Curve</h3>
                </div>
                {data?.equity_curve && data.equity_curve.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={data.equity_curve}>
                            <defs>
                                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="ts" stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)' }} />
                            <YAxis stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)' }} />
                            <Tooltip
                                contentStyle={{ background: 'var(--bg-tertiary)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 'var(--radius-md)' }}
                            />
                            <Area type="monotone" dataKey="value" stroke="#3b82f6" fillOpacity={1} fill="url(#equityGradient)" />
                        </AreaChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="text-center text-muted" style={{ padding: 'var(--spacing-2xl)' }}>No equity data available</div>
                )}
            </div>

            {/* Drawdown Chart */}
            <div className="card mb-lg">
                <div className="card-header">
                    <h3 className="card-title">Drawdown</h3>
                </div>
                {data?.drawdown && data.drawdown.length > 0 ? (
                    <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={data.drawdown}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="ts" stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)' }} />
                            <YAxis stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)' }} />
                            <Tooltip
                                contentStyle={{ background: 'var(--bg-tertiary)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 'var(--radius-md)' }}
                            />
                            <Line type="monotone" dataKey="drawdown" stroke="#ef4444" strokeWidth={2} />
                        </LineChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="text-center text-muted" style={{ padding: 'var(--spacing-xl)' }}>No drawdown data available</div>
                )}
            </div>

            {/* Open Trades */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Open Trades</h3>
                    <span className="badge badge-info">{data?.open_trades?.length || 0}</span>
                </div>
                {data?.open_trades && data.open_trades.length > 0 ? (
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Entry</th>
                                <th>Qty</th>
                                <th>Stop</th>
                                <th>Target</th>
                                <th>Probability</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.open_trades.map((trade) => (
                                <tr key={trade.id}>
                                    <td className="font-semibold">{trade.symbol}</td>
                                    <td>₹{trade.entry_price?.toFixed(2)}</td>
                                    <td>{trade.qty}</td>
                                    <td className="text-danger">₹{trade.stop?.toFixed(2)}</td>
                                    <td className="text-success">₹{trade.target?.toFixed(2)}</td>
                                    <td>
                                        <span className={`badge ${trade.probability >= 0.75 ? 'badge-success' : 'badge-warning'}`}>
                                            {(trade.probability * 100).toFixed(0)}%
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <div className="text-center text-muted" style={{ padding: 'var(--spacing-xl)' }}>No open trades</div>
                )}
            </div>

            {/* V3 Risk Controls Panel */}
            {stats?.engine_version === 'v2' && (
                <div className="grid grid-cols-3 mt-lg">
                    {/* Override Mode */}
                    <div className="card" style={{
                        background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1))',
                        border: '1px solid rgba(139, 92, 246, 0.3)'
                    }}>
                        <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            🎛️ Override Mode
                        </h3>
                        <div style={{ display: 'flex', gap: 'var(--spacing-sm)', flexWrap: 'wrap', marginTop: 'var(--spacing-md)' }}>
                            {['FULL', 'REDUCED', 'LONG_ONLY', 'PAPER_ONLY', 'KILLED'].map(mode => (
                                <button
                                    key={mode}
                                    className={`btn ${stats?.override_mode === mode ? 'btn-primary' : 'btn-outline'}`}
                                    style={{
                                        fontSize: '0.75rem',
                                        padding: '4px 12px',
                                        opacity: stats?.override_mode === mode ? 1 : 0.6
                                    }}
                                    onClick={() => {/* TODO: Implement mode change */ }}
                                >
                                    {mode === 'FULL' && '🟢'}
                                    {mode === 'REDUCED' && '🟡'}
                                    {mode === 'LONG_ONLY' && '📈'}
                                    {mode === 'PAPER_ONLY' && '📝'}
                                    {mode === 'KILLED' && '🛑'}
                                    {' '}{mode}
                                </button>
                            ))}
                        </div>
                        <p className="text-xs text-muted mt-md">
                            {stats?.override_mode === 'FULL' && 'Full trading enabled'}
                            {stats?.override_mode === 'REDUCED' && '50% position sizes'}
                            {stats?.override_mode === 'LONG_ONLY' && 'No short positions allowed'}
                            {stats?.override_mode === 'PAPER_ONLY' && 'Paper trading only'}
                            {stats?.override_mode === 'KILLED' && 'All trading stopped'}
                            {!stats?.override_mode && 'Full trading enabled (default)'}
                        </p>
                    </div>

                    {/* Capital Curve Feedback */}
                    <div className="card" style={{
                        background: stats?.risk_reduction > 0
                            ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(245, 158, 11, 0.1))'
                            : 'var(--bg-secondary)',
                        border: stats?.risk_reduction > 0 ? '1px solid rgba(239, 68, 68, 0.3)' : 'none'
                    }}>
                        <h3 className="card-title">📉 Drawdown Protection</h3>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'var(--spacing-md)' }}>
                            <div>
                                <div className="text-sm text-muted">Current Drawdown</div>
                                <div className="text-xl font-bold" style={{ color: (stats?.current_drawdown || 0) > 10 ? '#ef4444' : '#10b981' }}>
                                    {(stats?.current_drawdown || 0).toFixed(1)}%
                                </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div className="text-sm text-muted">Risk Multiplier</div>
                                <div className="text-xl font-bold" style={{ color: (stats?.risk_multiplier || 1) < 1 ? '#f59e0b' : '#10b981' }}>
                                    {((stats?.risk_multiplier || 1) * 100).toFixed(0)}%
                                </div>
                            </div>
                        </div>
                        {(stats?.risk_reduction || 0) > 0 && (
                            <div className="mt-md" style={{
                                padding: 'var(--spacing-sm)',
                                background: 'rgba(239, 68, 68, 0.2)',
                                borderRadius: 'var(--radius-sm)',
                                fontSize: '11px'
                            }}>
                                ⚠️ Risk reduced by {(stats?.risk_reduction || 0).toFixed(0)}% due to drawdown
                            </div>
                        )}
                    </div>

                    {/* Sector Exposure */}
                    <div className="card">
                        <h3 className="card-title">🏢 Sector Exposure</h3>
                        <div style={{ marginTop: 'var(--spacing-md)' }}>
                            {stats?.sector_exposure && Object.entries(stats.sector_exposure).length > 0 ? (
                                Object.entries(stats.sector_exposure).map(([sector, info]) => (
                                    <div key={sector} style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        padding: 'var(--spacing-xs) 0',
                                        borderBottom: '1px solid rgba(255,255,255,0.05)'
                                    }}>
                                        <span className="text-sm">{sector}</span>
                                        <span className="text-sm font-semibold">
                                            {info.count || 0} ({info.net_direction || 'N/A'})
                                        </span>
                                    </div>
                                ))
                            ) : (
                                <div className="text-center text-muted text-sm">No sector exposure</div>
                            )}
                        </div>
                        <div className="mt-md text-xs text-muted">
                            Max 2 positions per sector • 50% cap
                        </div>
                    </div>
                </div>
            )}

            {/* V3 Performance Stats (if available) */}
            {stats?.engine_version === 'v2' && stats?.regime_stats && (
                <div className="card mt-lg" style={{
                    background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(59, 130, 246, 0.05))'
                }}>
                    <h3 className="card-title">📊 Regime Performance (Decay-Weighted)</h3>
                    <div className="grid grid-cols-4 mt-md">
                        {Object.entries(stats.regime_stats).map(([regime, data]) => (
                            <div key={regime} style={{ textAlign: 'center', padding: 'var(--spacing-md)' }}>
                                <div className="text-xs text-muted">{regime}</div>
                                <div className="text-lg font-bold" style={{
                                    color: data.avg_r > 0 ? '#10b981' : data.avg_r < 0 ? '#ef4444' : '#9ca3af'
                                }}>
                                    {data.avg_r > 0 ? '+' : ''}{data.avg_r?.toFixed(2)}R
                                </div>
                                <div className="text-xs text-muted">
                                    {(data.win_rate * 100).toFixed(0)}% win • {data.count} trades
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

