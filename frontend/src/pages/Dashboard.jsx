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
        </div>
    );
}
