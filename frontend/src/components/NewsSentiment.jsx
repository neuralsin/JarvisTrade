import { useState, useEffect } from 'react';
import axios from 'axios';

/**
 * V3 News Sentiment Component
 * Displays real-time news sentiment for monitored stocks
 */
export default function NewsSentiment({ symbols = [] }) {
    const [sentiments, setSentiments] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (symbols.length > 0) {
            fetchSentiments();
        } else {
            setLoading(false);
        }
    }, [symbols]);

    const fetchSentiments = async () => {
        try {
            setLoading(true);
            const response = await axios.get('/api/v1/sentiment', {
                params: { symbols: symbols.join(',') }
            });
            setSentiments(response.data.sentiments || {});
        } catch (err) {
            console.error('Failed to fetch sentiment:', err);
            setError('News sentiment unavailable');
        } finally {
            setLoading(false);
        }
    };

    const getSentimentColor = (score) => {
        if (score > 0.2) return '#10b981'; // Green
        if (score < -0.2) return '#ef4444'; // Red
        return '#9ca3af'; // Gray
    };

    const getSentimentEmoji = (score) => {
        if (score > 0.3) return '🚀';
        if (score > 0.1) return '📈';
        if (score < -0.3) return '📉';
        if (score < -0.1) return '⚠️';
        return '➖';
    };

    if (loading) {
        return (
            <div className="card">
                <h3 className="card-title">📰 News Sentiment</h3>
                <div className="text-center" style={{ padding: 'var(--spacing-xl)' }}>
                    <div className="spinner" style={{ width: '24px', height: '24px' }}></div>
                </div>
            </div>
        );
    }

    if (error || Object.keys(sentiments).length === 0) {
        return (
            <div className="card">
                <h3 className="card-title">📰 News Sentiment</h3>
                <div className="text-center text-muted" style={{ padding: 'var(--spacing-xl)' }}>
                    {error || 'No sentiment data available'}
                </div>
            </div>
        );
    }

    return (
        <div className="card" style={{
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05))'
        }}>
            <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                📰 News Sentiment
                <button
                    className="btn btn-outline"
                    onClick={fetchSentiments}
                    style={{ fontSize: '0.7rem', padding: '4px 8px' }}
                >
                    🔄 Refresh
                </button>
            </h3>

            <div style={{ marginTop: 'var(--spacing-md)' }}>
                {Object.entries(sentiments).map(([symbol, data]) => (
                    <div
                        key={symbol}
                        style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            padding: 'var(--spacing-sm) var(--spacing-md)',
                            marginBottom: 'var(--spacing-sm)',
                            background: 'rgba(255,255,255,0.02)',
                            borderRadius: 'var(--radius-sm)',
                            border: `1px solid ${getSentimentColor(data.avg_sentiment)}22`
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                            <span className="font-semibold">{symbol}</span>
                            <span style={{ fontSize: '1.2rem' }}>{getSentimentEmoji(data.avg_sentiment)}</span>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-lg)' }}>
                            {/* Sentiment Score */}
                            <div style={{ textAlign: 'center' }}>
                                <div className="text-xs text-muted">Sentiment</div>
                                <div className="font-bold" style={{ color: getSentimentColor(data.avg_sentiment) }}>
                                    {data.avg_sentiment > 0 ? '+' : ''}{(data.avg_sentiment * 100).toFixed(0)}%
                                </div>
                            </div>

                            {/* News Count */}
                            <div style={{ textAlign: 'center' }}>
                                <div className="text-xs text-muted">News</div>
                                <div className="font-semibold">{data.news_count}</div>
                            </div>

                            {/* Bullish/Bearish */}
                            <div style={{ display: 'flex', gap: 'var(--spacing-xs)' }}>
                                <span style={{
                                    background: 'rgba(16, 185, 129, 0.2)',
                                    padding: '2px 6px',
                                    borderRadius: '4px',
                                    fontSize: '11px'
                                }}>
                                    🐂 {data.bullish_count}
                                </span>
                                <span style={{
                                    background: 'rgba(239, 68, 68, 0.2)',
                                    padding: '2px 6px',
                                    borderRadius: '4px',
                                    fontSize: '11px'
                                }}>
                                    🐻 {data.bearish_count}
                                </span>
                            </div>

                            {/* Confidence */}
                            <div style={{
                                width: '50px',
                                height: '6px',
                                background: 'rgba(255,255,255,0.1)',
                                borderRadius: '3px',
                                overflow: 'hidden'
                            }}>
                                <div style={{
                                    width: `${(data.confidence || 0) * 100}%`,
                                    height: '100%',
                                    background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                                    borderRadius: '3px'
                                }}></div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Top Headlines */}
            {Object.values(sentiments).some(s => s.top_headlines?.length > 0) && (
                <div style={{
                    marginTop: 'var(--spacing-md)',
                    padding: 'var(--spacing-sm)',
                    background: 'rgba(255,255,255,0.02)',
                    borderRadius: 'var(--radius-sm)'
                }}>
                    <div className="text-xs text-muted mb-sm">Recent Headlines</div>
                    {Object.entries(sentiments).map(([symbol, data]) => (
                        data.top_headlines?.slice(0, 2).map((headline, idx) => (
                            <div
                                key={`${symbol}-${idx}`}
                                className="text-xs"
                                style={{
                                    padding: 'var(--spacing-xs) 0',
                                    borderBottom: '1px solid rgba(255,255,255,0.03)',
                                    opacity: 0.8
                                }}
                            >
                                <span className="font-semibold" style={{ marginRight: '8px' }}>{symbol}:</span>
                                {headline.length > 80 ? headline.substring(0, 80) + '...' : headline}
                            </div>
                        ))
                    ))}
                </div>
            )}
        </div>
    );
}
