import { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

export default function Dashboard() {
    const [mode, setMode] = useState('paper');
    const [data, setData] = useState(null);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchDashboard();
        fetchStats();
    }, [mode]);

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
            <div className="grid grid-cols-4 mb-lg">
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
