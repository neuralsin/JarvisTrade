import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export default function SignalMonitor() {
    const [signals, setSignals] = useState([]);
    const [filteredSignals, setFilteredSignals] = useState([]);
    const [stockFilter, setStockFilter] = useState('ALL');
    const [actionFilter, setActionFilter] = useState('ALL');
    const [connected, setConnected] = useState(false);
    const ws = useRef(null);
    const [availableStocks, setAvailableStocks] = useState([]);

    useEffect(() => {
        fetchRecentSignals();
        connectWebSocket();
        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, []);

    useEffect(() => {
        let filtered = signals;

        if (stockFilter !== 'ALL') {
            filtered = filtered.filter(s => s.stock_symbol === stockFilter);
        }

        if (actionFilter !== 'ALL') {
            filtered = filtered.filter(s => s.action === actionFilter);
        }

        setFilteredSignals(filtered);
    }, [signals, stockFilter, actionFilter]);

    const fetchRecentSignals = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get(`${API_URL}/api/trading/signals/recent`, {
                headers: { Authorization: `Bearer ${token}` },
                params: { limit: 100 }
            });

            setSignals(response.data);

            const stocks = [...new Set(response.data.map(s => s.stock_symbol))];
            setAvailableStocks(stocks);
        } catch (error) {
            console.error('Failed to fetch signals:', error);
        }
    };

    const connectWebSocket = () => {
        const token = localStorage.getItem('token');
        ws.current = new WebSocket(`${WS_URL}/ws/signals/${token}`);

        ws.current.onopen = () => {
            console.log('WebSocket connected');
            setConnected(true);

            const pingInterval = setInterval(() => {
                if (ws.current?.readyState === WebSocket.OPEN) {
                    ws.current.send('ping');
                }
            }, 30000);

            ws.current.pingInterval = pingInterval;
        };

        ws.current.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.event === 'signal_generated') {
                    setSignals(prev => [data.data, ...prev].slice(0, 100));
                    setAvailableStocks(prev => {
                        const stocks = new Set(prev);
                        stocks.add(data.data.stock_symbol);
                        return Array.from(stocks);
                    });
                } else if (data.event === 'trade_executed') {
                    setSignals(prev => prev.map(s =>
                        s.id === data.data.signal_id
                            ? { ...s, trade_id: data.data.trade_id }
                            : s
                    ));
                }
            } catch (error) {
                console.error('WebSocket message error:', error);
            }
        };

        ws.current.onclose = () => {
            console.log('WebSocket disconnected');
            setConnected(false);

            if (ws.current?.pingInterval) {
                clearInterval(ws.current.pingInterval);
            }

            setTimeout(connectWebSocket, 5000);
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    };

    const getProbabilityBar = (probability) => {
        if (!probability) return null;

        const percent = Math.round(probability * 100);
        const width = `${percent}%`;

        let bgColor = 'var(--accent-warning)';
        if (percent >= 70) bgColor = 'var(--accent-success)';
        else if (percent >= 50) bgColor = 'var(--accent-warning)';
        else bgColor = 'var(--accent-danger)';

        return (
            <div style={{
                width: '100px',
                height: '8px',
                background: 'var(--bg-tertiary)',
                borderRadius: 'var(--radius-sm)'
            }}>
                <div style={{
                    width: width,
                    height: '100%',
                    background: bgColor,
                    borderRadius: 'var(--radius-sm)',
                    transition: 'width var(--transition-base)'
                }} />
            </div>
        );
    };

    const formatTime = (timestamp) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    const executeCount = filteredSignals.filter(s => s.action === 'EXECUTE').length;
    const rejectCount = filteredSignals.filter(s => s.action === 'REJECT').length;

    return (
        <div className="fade-in">
            {/* Header */}
            <div className="flex justify-between items-center mb-lg">
                <div>
                    <h1>Signal Monitor</h1>
                    <p className="text-muted text-sm">
                        Real-time signal tracking and decision logging
                    </p>
                </div>
                <div className="flex items-center gap-sm">
                    <div style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '50%',
                        background: connected ? 'var(--accent-success)' : 'var(--accent-danger)',
                        boxShadow: connected ? 'var(--shadow-glow-success)' : 'none'
                    }} />
                    <span className="font-semibold text-sm" style={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {connected ? 'LIVE' : 'DISCONNECTED'}
                    </span>
                </div>
            </div>

            {/* Filters */}
            <div className="card mb-lg">
                <div className="flex gap-lg items-end" style={{ flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '200px' }}>
                        <label className="form-label">Stock Symbol</label>
                        <select
                            value={stockFilter}
                            onChange={(e) => setStockFilter(e.target.value)}
                            className="form-input"
                        >
                            <option value="ALL">All Stocks</option>
                            {availableStocks.map(stock => (
                                <option key={stock} value={stock}>{stock}</option>
                            ))}
                        </select>
                    </div>

                    <div style={{ flex: 1, minWidth: '200px' }}>
                        <label className="form-label">Action Type</label>
                        <select
                            value={actionFilter}
                            onChange={(e) => setActionFilter(e.target.value)}
                            className="form-input"
                        >
                            <option value="ALL">All Actions</option>
                            <option value="EXECUTE">Execute Only</option>
                            <option value="REJECT">Reject Only</option>
                        </select>
                    </div>

                    <button onClick={fetchRecentSignals} className="btn btn-primary">
                        Refresh
                    </button>
                </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-3 mb-lg">
                <div className="card stat-card">
                    <div className="stat-value">{filteredSignals.length}</div>
                    <div className="stat-label">Total Signals</div>
                </div>

                <div className="card stat-card">
                    <div className="stat-value" style={{ color: 'var(--accent-success)' }}>{executeCount}</div>
                    <div className="stat-label">Executed</div>
                </div>

                <div className="card stat-card">
                    <div className="stat-value" style={{ color: 'var(--accent-danger)' }}>{rejectCount}</div>
                    <div className="stat-label">Rejected</div>
                </div>
            </div>

            {/* Signals Table */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Recent Signals</h3>
                    <span className="badge badge-info">{filteredSignals.length} signals</span>
                </div>
                <div style={{ overflowX: 'auto' }}>
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Stock</th>
                                <th>Direction</th>
                                <th>Regime</th>
                                <th>Probability</th>
                                <th>Action</th>
                                <th>Reason</th>
                                <th>Trade</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredSignals.map((signal) => (
                                <tr key={signal.id}>
                                    <td className="font-mono text-sm">
                                        {formatTime(signal.timestamp)}
                                    </td>
                                    <td>
                                        <span className="badge badge-info">{signal.stock_symbol}</span>
                                    </td>
                                    {/* V2: Direction */}
                                    <td>
                                        {signal.direction === 1 || signal.signal_type === 'BUY' ? (
                                            <span className="badge badge-success">📈 LONG</span>
                                        ) : signal.direction === 2 || signal.signal_type === 'SELL' ? (
                                            <span className="badge badge-danger">📉 SHORT</span>
                                        ) : (
                                            <span className="badge" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>➖ HOLD</span>
                                        )}
                                    </td>
                                    {/* V2: Regime */}
                                    <td>
                                        {signal.regime === 'TREND_STABLE' && (
                                            <span className="badge badge-success">📈 TREND</span>
                                        )}
                                        {signal.regime === 'TREND_VOLATILE' && (
                                            <span className="badge badge-warning">⚡ VOLATILE</span>
                                        )}
                                        {signal.regime === 'RANGE_QUIET' && (
                                            <span className="badge badge-info">➖ RANGE</span>
                                        )}
                                        {signal.regime === 'CHOP_PANIC' && (
                                            <span className="badge badge-danger">🛑 PANIC</span>
                                        )}
                                        {!signal.regime && (
                                            <span className="text-muted text-xs">V1</span>
                                        )}
                                    </td>
                                    <td>
                                        <div className="flex items-center gap-sm">
                                            {getProbabilityBar(signal.probability)}
                                            {signal.probability && (
                                                <span className="font-semibold text-sm">
                                                    {Math.round(signal.probability * 100)}%
                                                </span>
                                            )}
                                        </div>
                                    </td>
                                    <td>
                                        {signal.action === 'EXECUTE' ? (
                                            <span className="badge badge-success">✓ EXECUTE</span>
                                        ) : (
                                            <span className="badge badge-danger">✕ REJECT</span>
                                        )}
                                    </td>
                                    <td className="text-muted text-sm" style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {signal.reason || '-'}
                                    </td>
                                    <td>
                                        {signal.trade_id ? (
                                            <a
                                                href={`/trades?id=${signal.trade_id}`}
                                                className="text-primary font-medium"
                                                style={{ textDecoration: 'none' }}
                                            >
                                                View Trade →
                                            </a>
                                        ) : (
                                            <span className="text-muted">-</span>
                                        )}
                                    </td>
                                </tr>
                            ))}

                            {filteredSignals.length === 0 && (
                                <tr>
                                    <td colSpan={8} style={{ padding: 'var(--spacing-2xl)', textAlign: 'center' }}>
                                        <div className="text-muted" style={{ fontSize: '3rem', marginBottom: 'var(--spacing-md)' }}>📄</div>
                                        <p className="font-medium text-muted">No signals yet</p>
                                        <p className="text-muted text-sm mt-sm">Signals will appear here in real-time as they are generated</p>
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
