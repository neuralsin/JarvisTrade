import { useState, useEffect } from 'react';
import axios from 'axios';

export default function ManageStocks() {
    const [stocks, setStocks] = useState([]);
    const [newSymbol, setNewSymbol] = useState('');
    const [dataPeriod, setDataPeriod] = useState('60d'); // Default 60 days
    const [validation, setValidation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [validating, setValidating] = useState(false);

    useEffect(() => {
        fetchStocks();
    }, []);

    const fetchStocks = async () => {
        try {
            const response = await axios.get('/api/v1/instruments/');
            setStocks(response.data);
        } catch (error) {
            console.error('Failed to fetch stocks:', error);
        }
    };

    const validateSymbol = async () => {
        if (!newSymbol || newSymbol.length < 2) {
            setValidation(null);
            return;
        }

        setValidating(true);
        try {
            const response = await axios.post('/api/v1/instruments/validate', {
                symbol: newSymbol.toUpperCase(),
                exchange: 'NS'
            });
            setValidation(response.data);
        } catch (error) {
            setValidation({
                valid: false,
                error: error.response?.data?.detail || error.message
            });
        } finally {
            setValidating(false);
        }
    };

    const addStock = async () => {
        if (!validation?.valid) return;

        setLoading(true);
        try {
            const response = await axios.post('/api/v1/instruments/add', {
                symbol: validation.symbol,
                name: validation.name,
                exchange: 'NS',
                data_period: dataPeriod // Send the period
            });

            alert(response.data.message || `Added ${validation.symbol} successfully!`);
            setNewSymbol('');
            setDataPeriod('60d'); // Reset to default
            setValidation(null);
            fetchStocks();
        } catch (error) {
            alert(`Failed to add stock: ${error.response?.data?.detail || error.message}`);
        } finally {
            setLoading(false);
        }
    };

    const removeStock = async (id, symbol) => {
        if (!confirm(`Remove ${symbol}? This will delete all associated data.`)) return;

        try {
            await axios.delete(`/api/v1/instruments/${id}`);
            alert(`Removed ${symbol}`);
            fetchStocks();
        } catch (error) {
            alert(`Failed to remove: ${error.response?.data?.detail || error.message}`);
        }
    };

    return (
        <div className="fade-in">
            <div className="mb-lg">
                <h1>📊 Manage Stocks</h1>
                <p className="text-muted">Add or remove stocks for paper trading and model training</p>
            </div>

            {/* Add Stock Form */}
            <div className="card mb-lg">
                <h3 className="card-title mb-md">Add New Stock</h3>

                <div style={{ display: 'flex', gap: 'var(--spacing-md)', alignItems: 'flex-end', marginBottom: 'var(--spacing-md)' }}>
                    <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
                        <label className="form-label">NSE Stock Symbol</label>
                        <input
                            type="text"
                            className="form-input"
                            value={newSymbol}
                            onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                            onBlur={validateSymbol}
                            placeholder="e.g., RELIANCE, TCS, INFY"
                            style={{ textTransform: 'uppercase' }}
                        />
                    </div>
                    <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
                        <label className="form-label">Historical Data Period</label>
                        <select
                            className="form-input"
                            value={dataPeriod}
                            onChange={(e) => setDataPeriod(e.target.value)}
                        >
                            <option value="60d">📅 Last 60 Days (15m candles) - Fastest</option>
                            <option value="1y">📅 Last 1 Year (1d candles) - ~252 days</option>
                            <option value="2y">📅 Last 2 Years (1d candles) - ~504 days</option>
                            <option value="5y">📅 Last 5 Years (1d candles) - ~1260 days</option>
                        </select>
                        <div className="text-xs text-muted" style={{ marginTop: 'var(--spacing-xs)' }}>
                            💡 More data = Better models but slower fetch
                        </div>
                    </div>
                    <button
                        className="btn btn-primary"
                        onClick={addStock}
                        disabled={!validation?.valid || loading}
                    >
                        {loading ? 'Adding...' : '➕ Add Stock'}
                    </button>
                </div>

                {/* Validation Status */}
                {validating && (
                    <div className="card" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                            <div className="spinner" style={{ width: '16px', height: '16px' }}></div>
                            <span>Validating symbol...</span>
                        </div>
                    </div>
                )}

                {!validating && validation && (
                    <div className={`card ${validation.valid ? '' : ''}`} style={{
                        background: validation.valid
                            ? 'rgba(34, 197, 94, 0.1)'
                            : 'rgba(239, 68, 68, 0.1)',
                        border: validation.valid
                            ? '1px solid rgba(34, 197, 94, 0.3)'
                            : '1px solid rgba(239, 68, 68, 0.3)'
                    }}>
                        {validation.valid ? (
                            <div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-xs)' }}>
                                    <span style={{ fontSize: '1.5rem' }}>✓</span>
                                    <div>
                                        <div className="font-bold">{validation.name}</div>
                                        <div className="text-sm text-muted">{validation.symbol} · NSE</div>
                                    </div>
                                </div>
                                {validation.sector && (
                                    <div className="text-sm text-muted">
                                        {validation.sector} {validation.industry ? `· ${validation.industry}` : ''}
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                                <span style={{ fontSize: '1.5rem' }}>❌</span>
                                <span>{validation.error}</span>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Stock List */}
            <div className="card">
                <h3 className="card-title mb-md">Current Stocks ({stocks.length})</h3>

                {stocks.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: 'var(--spacing-xxl)', color: 'var(--text-muted)' }}>
                        <div style={{ fontSize: '3rem', marginBottom: 'var(--spacing-md)' }}>📈</div>
                        <p>No stocks added yet. Add your first stock above!</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-3" style={{ gap: 'var(--spacing-md)' }}>
                        {stocks.map(stock => (
                            <div key={stock.id} className="card" style={{ background: 'var(--bg-tertiary)' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <div style={{ flex: 1 }}>
                                        <div className="font-bold text-lg" style={{ marginBottom: 'var(--spacing-xs)' }}>
                                            {stock.symbol}
                                        </div>
                                        <div className="text-sm text-muted" style={{ marginBottom: 'var(--spacing-xs)' }}>
                                            {stock.name}
                                        </div>
                                        <div className="text-xs" style={{ color: 'var(--accent-primary)' }}>
                                            {stock.exchange} · {stock.instrument_type}
                                        </div>
                                    </div>
                                    <button
                                        className="btn btn-sm"
                                        onClick={() => removeStock(stock.id, stock.symbol)}
                                        style={{
                                            background: 'rgba(239, 68, 68, 0.1)',
                                            border: '1px solid rgba(239, 68, 68, 0.3)',
                                            color: 'var(--accent-danger)'
                                        }}
                                    >
                                        🗑️
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
