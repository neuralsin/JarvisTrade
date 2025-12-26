import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export default function TradingControls() {
    const [status, setStatus] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);
    const [selectedModels, setSelectedModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [actionLoading, setActionLoading] = useState(false);

    // Fetch trading status and models on mount
    useEffect(() => {
        fetchTradingStatus();
        fetchAvailableModels();
        fetchSelectedModels();
        const interval = setInterval(fetchSelectedModels, 30000); // Refresh every 30s
        return () => clearInterval(interval);
    }, []);

    const fetchTradingStatus = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get(`${API_URL}/api/trading/status`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setStatus(response.data);
        } catch (error) {
            console.error('Failed to fetch trading status:', error);
        }
    };

    const fetchAvailableModels = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get(`${API_URL}/api/trading/models/available`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setAvailableModels(response.data);
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch models:', error);
            setLoading(false);
        }
    };

    const fetchSelectedModels = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get(`${API_URL}/api/trading/models/selected`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setSelectedModels(response.data.selected_models || []);
        } catch (error) {
            console.error('Failed to fetch selected models:', error);
        }
    };

    const togglePaperTrading = async (enabled) => {
        setActionLoading(true);
        try {
            const token = localStorage.getItem('token');
            await axios.post(
                `${API_URL}/api/trading/paper/toggle`,
                { enabled },
                { headers: { Authorization: `Bearer ${token}` } }
            );
            setStatus({ ...status, paper_trading_enabled: enabled });
        } catch (error) {
            console.error('Failed to toggle paper trading:', error);
            alert('Failed to toggle paper trading');
        }
        setActionLoading(false);
    };

    const toggleModelSelection = async (modelId, isSelected) => {
        try {
            const token = localStorage.getItem('token');
            if (isSelected) {
                await axios.delete(`${API_URL}/api/trading/models/select/${modelId}`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
            } else {
                await axios.post(
                    `${API_URL}/api/trading/models/select`,
                    { model_id: modelId },
                    { headers: { Authorization: `Bearer ${token}` } }
                );
            }
            fetchAvailableModels();
            fetchSelectedModels();
            fetchTradingStatus();
        } catch (error) {
            console.error('Failed to toggle model:', error);
            alert('Failed to update model selection');
        }
    };

    const selectAllModels = async () => {
        setActionLoading(true);
        try {
            const token = localStorage.getItem('token');
            await axios.post(
                `${API_URL}/api/trading/models/select-all`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );
            fetchAvailableModels();
            fetchSelectedModels();
            fetchTradingStatus();
        } catch (error) {
            console.error('Failed to select all:', error);
            alert('Failed to select all models');
        }
        setActionLoading(false);
    };

    const clearAllSelections = async () => {
        setActionLoading(true);
        try {
            const token = localStorage.getItem('token');
            await axios.delete(`${API_URL}/api/trading/models/clear`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            fetchAvailableModels();
            fetchSelectedModels();
            fetchTradingStatus();
        } catch (error) {
            console.error('Failed to clear selections:', error);
            alert('Failed to clear selections');
        }
        setActionLoading(false);
    };

    const checkNow = async () => {
        setActionLoading(true);
        try {
            const token = localStorage.getItem('token');
            await axios.post(
                `${API_URL}/api/trading/check-now`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );
            alert('Signal check triggered! Monitor the Signal Monitor for results.');
        } catch (error) {
            console.error('Failed to trigger check:', error);
            alert('Failed to trigger signal check');
        }
        setActionLoading(false);
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6 max-w-6xl mx-auto">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-gray-900">Trading Controls</h1>
                <p className="text-gray-600 mt-2">
                    Configure paper trading and select models to run
                </p>
            </div>

            {/* Paper Trading Toggle */}
            <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Paper Trading</h2>
                <div className="flex items-center justify-between">
                    <div>
                        <p className="font-medium text-gray-900">Master Switch</p>
                        <p className="text-sm text-gray-600">
                            Enable or disable all paper trading activity
                        </p>
                    </div>
                    <button
                        onClick={() => togglePaperTrading(!status?.paper_trading_enabled)}
                        disabled={actionLoading}
                        className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${status?.paper_trading_enabled ? 'bg-green-600' : 'bg-gray-300'
                            } ${actionLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                        <span
                            className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${status?.paper_trading_enabled ? 'translate-x-7' : 'translate-x-1'
                                }`}
                        />
                    </button>
                </div>
                {status?.paper_trading_enabled && (
                    <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                        <div className="flex items-center gap-2">
                            <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            <span className="font-medium text-green-800">
                                Paper Trading Active
                            </span>
                        </div>
                    </div>
                )}
            </div>

            {/* Active Models */}
            {status?.selected_model_count > 0 && (
                <div className="bg-white rounded-lg shadow p-6">
                    <h2 className="text-xl font-semibold mb-4">Active Models</h2>
                    <p className="text-sm text-gray-600 mb-4">
                        {status.selected_model_count} model(s) running on{' '}
                        {status.selected_stocks.length} stock(s)
                    </p>
                    <div className="flex flex-wrap gap-2">
                        {status.selected_stocks.map((stock) => (
                            <span
                                key={stock}
                                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
                            >
                                {stock}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Quick Actions */}
            <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
                <div className="flex flex-wrap gap-3">
                    <button
                        onClick={selectAllModels}
                        disabled={actionLoading}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Select All Models
                    </button>
                    <button
                        onClick={clearAllSelections}
                        disabled={actionLoading}
                        className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Clear All
                    </button>
                    <button
                        onClick={checkNow}
                        disabled={actionLoading}
                        className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Check Signals Now
                    </button>
                </div>
            </div>

            {/* Model Selection */}
            <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4">Model Selection</h2>
                <div className="space-y-6">
                    {availableModels.map((stockGroup) => (
                        <div key={stockGroup.stock_symbol} className="border rounded-lg p-4">
                            <h3 className="font-bold text-lg mb-3 text-gray-900">{stockGroup.stock_symbol}</h3>
                            <div className="space-y-2">
                                {stockGroup.models.map((model) => (
                                    <div
                                        key={model.id}
                                        className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                                    >
                                        <div className="flex items-center gap-3">
                                            <input
                                                type="checkbox"
                                                checked={model.is_selected}
                                                onChange={() => toggleModelSelection(model.id, model.is_selected)}
                                                disabled={!model.is_active}
                                                className="h-5 w-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                            />
                                            <div>
                                                <p className="font-medium text-gray-900">{model.name}</p>
                                                <p className="text-sm text-gray-600">
                                                    {model.model_type.toUpperCase()} •{' '}
                                                    {model.is_active ? (
                                                        <span className="text-green-600 font-medium">Active</span>
                                                    ) : (
                                                        <span className="text-gray-400">Inactive</span>
                                                    )}
                                                </p>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            {model.metrics?.auc_roc && (
                                                <p className="text-sm text-gray-700">
                                                    AUC: <span className="font-semibold">{model.metrics.auc_roc.toFixed(3)}</span>
                                                </p>
                                            )}
                                            {model.metrics?.accuracy && (
                                                <p className="text-sm text-gray-600">
                                                    Acc: {(model.metrics.accuracy * 100).toFixed(1)}%
                                                </p>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}

                    {availableModels.length === 0 && (
                        <div className="text-center py-12 text-gray-500">
                            <svg className="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            <p className="text-lg font-medium">No models available</p>
                            <p className="text-sm mt-1">Train models first in the Models section</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
