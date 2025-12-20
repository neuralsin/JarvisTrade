import { useState, useEffect } from 'react';
import axios from 'axios';

export default function Trades() {
    const [mode, setMode] = useState('paper');
    const [status, setStatus] = useState('');
    const [trades, setTrades] = useState([]);
    const [selectedTrade, setSelectedTrade] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchTrades();
    }, [mode, status]);

    const fetchTrades = async () => {
        try {
            const params = new URLSearchParams({ mode });
            if (status) params.append('status', status);

            const response = await axios.get(`/api/v1/trades?${params.toString()}`);
            setTrades(response.data.trades);
        } catch (error) {
            console.error('Failed to fetch trades:', error);
        } finally {
            setLoading(false);
        }
    };

    const fetchTradeDetails = async (tradeId) => {
        try {
            const response = await axios.get(`/api/v1/trades/${tradeId}`);
            setSelectedTrade(response.data);
        } catch (error) {
            console.error('Failed to fetch trade details:', error);
        }
    };

    const formatDate = (dateStr) => {
        if (!dateStr) return 'N/A';
        return new Date(dateStr).toLocaleString();
    };

    if (loading) {
        return <div className="flex justify-center items-center" style={{ height: '400px' }}><div className="spinner"></div></div>;
    }

    return (
        <div className="fade-in">
            <h1 className="mb-lg">Trade History</h1>

            {/* Filters */}
            <div className="card mb-md">
                <div className="flex gap-md items-center">
                    <div>
                        <label className="form-label">Mode</label>
                        <select className="form-input" value={mode} onChange={(e) => setMode(e.target.value)}>
                            <option value="paper">Paper</option>
                            <option value="live">Live</option>
                        </select>
                    </div>
                    <div>
                        <label className="form-label">Status</label>
                        <select className="form-input" value={status} onChange={(e) => setStatus(e.target.value)}>
                            <option value="">All</option>
                            <option value="open">Open</option>
                            <option value="closed">Closed</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Trades Table */}
            <div className="card">
                {trades.length === 0 ? (
                    <div className="text-center text-muted" style={{ padding: 'var(--spacing-2xl)' }}>
                        No trades found
                    </div>
                ) : (
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>Qty</th>
                                <th>P&L</th>
                                <th>Status</th>
                                <th>Probability</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {trades.map((trade) => (
                                <tr key={trade.id}>
                                    <td className="font-semibold">{trade.symbol}</td>
                                    <td>
                                        <div>₹{trade.entry_price?.toFixed(2)}</div>
                                        <div className="text-xs text-muted">{formatDate(trade.entry_ts)}</div>
                                    </td>
                                    <td>
                                        {trade.exit_price ? (
                                            <>
                                                <div>₹{trade.exit_price.toFixed(2)}</div>
                                                <div className="text-xs text-muted">{formatDate(trade.exit_ts)}</div>
                                            </>
                                        ) : (
                                            <span className="text-muted">—</span>
                                        )}
                                    </td>
                                    <td>{trade.qty}</td>
                                    <td>
                                        {trade.pnl !== null ? (
                                            <span className={trade.pnl >= 0 ? 'text-success font-semibold' : 'text-danger font-semibold'}>
                                                ₹{trade.pnl.toFixed(2)}
                                            </span>
                                        ) : (
                                            <span className="text-muted">—</span>
                                        )}
                                    </td>
                                    <td>
                                        <span className={`badge ${trade.status === 'open' ? 'badge-info' : 'badge-success'}`}>
                                            {trade.status}
                                        </span>
                                    </td>
                                    <td>
                                        {trade.probability && (
                                            <span className={`badge ${trade.probability >= 0.75 ? 'badge-success' : 'badge-warning'}`}>
                                                {(trade.probability * 100).toFixed(0)}%
                                            </span>
                                        )}
                                    </td>
                                    <td>
                                        <button
                                            className="btn btn-sm btn-outline"
                                            onClick={() => fetchTradeDetails(trade.id)}
                                        >
                                            Details
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Trade Details Modal */}
            {selectedTrade && (
                <div className="modal-overlay" onClick={() => setSelectedTrade(null)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '800px' }}>
                        <h3 className="mb-lg">Trade Details: {selectedTrade.symbol}</h3>

                        <div className="grid grid-cols-2 gap-md mb-lg">
                            <div>
                                <div className="text-muted text-sm">Trade ID</div>
                                <div className="text-sm font-mono">{selectedTrade.trade_id}</div>
                            </div>
                            <div>
                                <div className="text-muted text-sm">Status</div>
                                <span className={`badge ${selectedTrade.status === 'open' ? 'badge-info' : 'badge-success'}`}>
                                    {selectedTrade.status}
                                </span>
                            </div>
                            <div>
                                <div className="text-muted text-sm">Entry Price</div>
                                <div className="text-lg font-semibold">₹{selectedTrade.entry_price?.toFixed(2)}</div>
                            </div>
                            <div>
                                <div className="text-muted text-sm">Exit Price</div>
                                <div className="text-lg font-semibold">
                                    {selectedTrade.exit_price ? `₹${selectedTrade.exit_price.toFixed(2)}` : 'N/A'}
                                </div>
                            </div>
                            <div>
                                <div className="text-muted text-sm">Stop Loss</div>
                                <div className="text-danger">₹{selectedTrade.stop_price?.toFixed(2)}</div>
                            </div>
                            <div>
                                <div className="text-muted text-sm">Target</div>
                                <div className="text-success">₹{selectedTrade.target_price?.toFixed(2)}</div>
                            </div>
                            <div>
                                <div className="text-muted text-sm">Quantity</div>
                                <div>{selectedTrade.qty}</div>
                            </div>
                            <div>
                                <div className="text-muted text-sm">P&L</div>
                                <div className={`text-lg font-bold ${selectedTrade.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                                    {selectedTrade.pnl ? `₹${selectedTrade.pnl.toFixed(2)}` : 'Open'}
                                </div>
                            </div>
                            <div>
                                <div className="text-muted text-sm">Probability</div>
                                <div>{(selectedTrade.probability * 100).toFixed(1)}%</div>
                            </div>
                            <div>
                                <div className="text-muted text-sm">Model</div>
                                <div className="text-sm">{selectedTrade.model?.name || 'N/A'}</div>
                            </div>
                        </div>

                        {/* Logs */}
                        {selectedTrade.logs && selectedTrade.logs.length > 0 && (
                            <div className="mt-lg">
                                <h4 className="mb-md">Trade Logs</h4>
                                <div style={{ background: 'var(--bg-tertiary)', padding: 'var(--spacing-md)', borderRadius: 'var(--radius-md)', maxHeight: '200px', overflowY: 'auto' }}>
                                    {selectedTrade.logs.map((log, idx) => (
                                        <div key={idx} className="text-sm" style={{ marginBottom: 'var(--spacing-sm)' }}>
                                            <span className="text-muted">[{new Date(log.ts).toLocaleTimeString()}]</span>{' '}
                                            <span className={log.level === 'ERROR' ? 'text-danger' : ''}>{log.text}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        <div className="flex justify-end mt-lg">
                            <button className="btn btn-primary" onClick={() => setSelectedTrade(null)}>
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
