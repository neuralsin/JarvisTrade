import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
        const interval = setInterval(fetchSelectedModels, 30000);
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
            <div className="flex justify-center items-center" style={{ height: '400px' }}>
                <div className="spinner"></div>
            </div>
        );
    }

    return (
        <div className="fade-in">
            {/* Header */}
            <div className="flex justify-between items-center mb-lg">
                <div>
                    <h1>Trading Controls</h1>
                    <p className="text-muted text-sm">
                        Configure paper trading and select models to run
                    </p>
                </div>
            </div>

            {/* Paper Trading Toggle Card */}
            <div className="card mb-lg">
                <div className="card-header">
                    <h3 className="card-title">Paper Trading</h3>
                    <div
                        className="toggle-switch"
                        onClick={() => !actionLoading && togglePaperTrading(!status?.paper_trading_enabled)}
                        style={{
                            width: '56px',
                            height: '32px',
                            borderRadius: '16px',
                            background: status?.paper_trading_enabled ? 'var(--accent-success)' : 'var(--bg-tertiary)',
                            cursor: actionLoading ? 'not-allowed' : 'pointer',
                            opacity: actionLoading ? 0.5 : 1,
                            transition: 'background var(--transition-base)',
                            position: 'relative'
                        }}
                    >
                        <div style={{
                            width: '24px',
                            height: '24px',
                            borderRadius: '50%',
                            background: 'white',
                            position: 'absolute',
                            top: '4px',
                            left: status?.paper_trading_enabled ? '28px' : '4px',
                            transition: 'left var(--transition-base)',
                            boxShadow: 'var(--shadow-sm)'
                        }} />
                    </div>
                </div>
                <div className="flex items-center gap-md">
                    <div>
                        <p className="font-semibold">Master Switch</p>
                        <p className="text-muted text-sm">
                            Enable or disable all paper trading activity
                        </p>
                    </div>
                </div>
                {status?.paper_trading_enabled && (
                    <div className="mt-md" style={{
                        padding: 'var(--spacing-md)',
                        background: 'rgba(16, 185, 129, 0.1)',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        borderRadius: 'var(--radius-md)'
                    }}>
                        <div className="flex items-center gap-sm">
                            <span className="text-success">✓</span>
                            <span className="font-medium text-success">Paper Trading Active</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Active Models Summary */}
            {status?.selected_model_count > 0 && (
                <div className="card mb-lg">
                    <div className="card-header">
                        <h3 className="card-title">Active Models</h3>
                        <span className="badge badge-info">{status.selected_model_count}</span>
                    </div>
                    <p className="text-muted text-sm mb-md">
                        {status.selected_model_count} model(s) running on {status.selected_stocks.length} stock(s)
                    </p>
                    <div className="flex gap-sm" style={{ flexWrap: 'wrap' }}>
                        {status.selected_stocks.map((stock) => (
                            <span key={stock} className="badge badge-info">
                                {stock}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Quick Actions */}
            <div className="card mb-lg">
                <div className="card-header">
                    <h3 className="card-title">Quick Actions</h3>
                </div>
                <div className="flex gap-md" style={{ flexWrap: 'wrap' }}>
                    <button
                        onClick={selectAllModels}
                        disabled={actionLoading}
                        className="btn btn-primary"
                    >
                        Select All Models
                    </button>
                    <button
                        onClick={clearAllSelections}
                        disabled={actionLoading}
                        className="btn btn-outline"
                    >
                        Clear All
                    </button>
                    <button
                        onClick={checkNow}
                        disabled={actionLoading}
                        className="btn btn-success"
                    >
                        ▶ Check Signals Now
                    </button>
                </div>
            </div>

            {/* Model Selection */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Model Selection</h3>
                </div>
                <div className="flex flex-col gap-lg">
                    {availableModels.map((stockGroup) => (
                        <div key={stockGroup.stock_symbol} style={{
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            borderRadius: 'var(--radius-md)',
                            padding: 'var(--spacing-md)'
                        }}>
                            <h4 className="mb-md" style={{ color: 'var(--accent-primary)' }}>
                                {stockGroup.stock_symbol}
                            </h4>
                            <div className="flex flex-col gap-sm">
                                {stockGroup.models.map((model) => (
                                    <div
                                        key={model.id}
                                        className="flex items-center justify-between"
                                        style={{
                                            padding: 'var(--spacing-md)',
                                            background: 'var(--bg-tertiary)',
                                            borderRadius: 'var(--radius-sm)',
                                            transition: 'background var(--transition-fast)'
                                        }}
                                    >
                                        <div className="flex items-center gap-md">
                                            <input
                                                type="checkbox"
                                                checked={model.is_selected}
                                                onChange={() => toggleModelSelection(model.id, model.is_selected)}
                                                disabled={!model.is_active}
                                                style={{
                                                    width: '20px',
                                                    height: '20px',
                                                    accentColor: 'var(--accent-primary)',
                                                    cursor: model.is_active ? 'pointer' : 'not-allowed'
                                                }}
                                            />
                                            <div>
                                                <p className="font-medium">{model.name}</p>
                                                <p className="text-muted text-xs">
                                                    {model.model_type.toUpperCase()} • {' '}
                                                    {model.is_active ? (
                                                        <span className="text-success">Active</span>
                                                    ) : (
                                                        <span className="text-muted">Inactive</span>
                                                    )}
                                                </p>
                                            </div>
                                        </div>
                                        <div style={{ textAlign: 'right' }}>
                                            {model.metrics?.auc_roc && (
                                                <p className="text-sm">
                                                    AUC: <span className="font-semibold">{model.metrics.auc_roc.toFixed(3)}</span>
                                                </p>
                                            )}
                                            {model.metrics?.accuracy && (
                                                <p className="text-muted text-xs">
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
                        <div className="text-center" style={{ padding: 'var(--spacing-2xl)' }}>
                            <div className="text-muted mb-md" style={{ fontSize: '3rem' }}>⚠️</div>
                            <p className="font-medium text-muted">No models available</p>
                            <p className="text-muted text-sm mt-sm">Train models first in the Models section</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
