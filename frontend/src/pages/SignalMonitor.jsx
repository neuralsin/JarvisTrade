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

    // Fetch recent signals on mount
    useEffect(() => {
        fetchRecentSignals();
        connectWebSocket();
        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, []);

    // Filter signals when filters change
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

            // Extract unique stocks
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

            // Send ping every 30s to keep alive
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
                    // Add new signal to top of list
                    setSignals(prev => [data.data, ...prev].slice(0, 100));
                    setAvailableStocks(prev => {
                        const stocks = new Set(prev);
                        stocks.add(data.data.stock_symbol);
                        return Array.from(stocks);
                    });
                } else if (data.event === 'trade_executed') {
                    // Update signal with trade_id
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

            // Clear ping interval
            if (ws.current?.pingInterval) {
                clearInterval(ws.current.pingInterval);
            }

            // Reconnect after 5s
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

        let bgColor = 'bg-gray-400';
        if (percent >= 70) bgColor = 'bg-green-500';
        else if (percent >= 50) bgColor = 'bg-yellow-500';
        else bgColor = 'bg-red-500';

        return (
            <div className="w-24 bg-gray-200 rounded-full h-2">
                <div className={`${bgColor} h-2 rounded-full transition-all duration-300`} style={{ width }} />
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
        <div className="p-6 space-y-6 max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Signal Monitor</h1>
                    <p className="text-gray-600 mt-2">
                        Real-time signal tracking and decision logging
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-sm font-semibold uppercase tracking-wide">
                        {connected ? 'LIVE' : 'DISCONNECTED'}
                    </span>
                </div>
            </div>

            {/* Filters */}
            <div className="bg-white rounded-lg shadow p-6">
                <div className="flex gap-4 items-end">
                    <div className="flex-1">
                        <label className="text-sm font-medium text-gray-700 mb-2 block">Stock Symbol</label>
                        <select
                            value={stockFilter}
                            onChange={(e) => setStockFilter(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="ALL">All Stocks</option>
                            {availableStocks.map(stock => (
                                <option key={stock} value={stock}>{stock}</option>
                            ))}
                        </select>
                    </div>

                    <div className="flex-1">
                        <label className="text-sm font-medium text-gray-700 mb-2 block">Action Type</label>
                        <select
                            value={actionFilter}
                            onChange={(e) => setActionFilter(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="ALL">All Actions</option>
                            <option value="EXECUTE">Execute Only</option>
                            <option value="REJECT">Reject Only</option>
                        </select>
                    </div>

                    <button
                        onClick={fetchRecentSignals}
                        className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                        Refresh
                    </button>
                </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-3 gap-4">
                <div className="bg-white rounded-lg shadow p-6">
                    <div className="text-center">
                        <div className="text-4xl font-bold text-gray-900">{filteredSignals.length}</div>
                        <div className="text-sm text-gray-600 mt-2 font-medium">Total Signals</div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="text-center">
                        <div className="text-4xl font-bold text-green-600">{executeCount}</div>
                        <div className="text-sm text-gray-600 mt-2 font-medium">Executed</div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                    <div className="text-center">
                        <div className="text-4xl font-bold text-red-600">{rejectCount}</div>
                        <div className="text-sm text-gray-600 mt-2 font-medium">Rejected</div>
                    </div>
                </div>
            </div>

            {/* Signals Table */}
            <div className="bg-white rounded-lg shadow overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50 border-b border-gray-200">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Time</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Stock</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Model</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Probability</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Action</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Reason</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">Trade</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {filteredSignals.map((signal) => (
                                <tr key={signal.id} className="hover:bg-gray-50 transition-colors">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                                        {formatTime(signal.timestamp)}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                            {signal.stock_symbol}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                                        {signal.model_name}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="flex items-center gap-3">
                                            {getProbabilityBar(signal.probability)}
                                            {signal.probability && (
                                                <span className="text-sm font-semibold text-gray-700">
                                                    {Math.round(signal.probability * 100)}%
                                                </span>
                                            )}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        {signal.action === 'EXECUTE' ? (
                                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-green-100 text-green-800">
                                                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                </svg>
                                                EXECUTE
                                            </span>
                                        ) : (
                                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800">
                                                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                                </svg>
                                                REJECT
                                            </span>
                                        )}
                                    </td>
                                    <td className="px-6 py-4 text-sm text-gray-600 max-w-xs truncate">
                                        {signal.reason || '-'}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                                        {signal.trade_id ? (
                                            <a
                                                href={`/trades?id=${signal.trade_id}`}
                                                className="text-blue-600 hover:text-blue-800 font-medium hover:underline"
                                            >
                                                View Trade →
                                            </a>
                                        ) : (
                                            <span className="text-gray-400">-</span>
                                        )}
                                    </td>
                                </tr>
                            ))}

                            {filteredSignals.length === 0 && (
                                <tr>
                                    <td colSpan={7} className="px-6 py-16 text-center">
                                        <div className="flex flex-col items-center justify-center text-gray-500">
                                            <svg className="w-16 h-16 mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                            </svg>
                                            <p className="text-lg font-medium">No signals yet</p>
                                            <p className="text-sm mt-1">Signals will appear here in real-time as they are generated</p>
                                        </div>
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
