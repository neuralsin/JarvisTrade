import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function SentimentCard({ symbol }) {
    const [loading, setLoading] = useState(false);
    const [sentiment, setSentiment] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (symbol) {
            fetchSentiment();
        }
    }, [symbol]);

    const fetchSentiment = async () => {
        setLoading(true);
        setError(null);
        try {
            // Check if we need to filter by symbol or getting all (though card implies specific symbol)
            // The API we implemented: GET /api/v1/sentiment/{symbol}
            const response = await axios.get(`/api/v1/sentiment/${symbol}`);

            // The API returns { symbol, instrument_id, data: [...] }
            // We want the most recent data point for the summary
            if (response.data.data && response.data.data.length > 0) {
                setSentiment(response.data.data[0]); // Get latest
            } else {
                setSentiment(null); // No data found
            }

        } catch (err) {
            console.error('Failed to fetch sentiment:', err);
            setError('Could not load sentiment data');
        } finally {
            setLoading(false);
        }
    };

    const triggerFetch = async () => {
        setLoading(true);
        try {
            await axios.post('/api/v1/sentiment/fetch');
            // Poll for update after a few seconds? Or let the user click refresh again?
            // For now, simple alert and reload after 5s
            setTimeout(fetchSentiment, 5000);
        } catch (err) {
            alert('Failed to trigger analysis: ' + (err.response?.data?.error || err.message));
            setLoading(false);
        }
    };

    if (!symbol) return null;

    const getScoreColor = (score) => {
        if (score > 0.2) return '#10b981'; // Green
        if (score < -0.2) return '#ef4444'; // Red
        return '#f59e0b'; // Amber/Neutral
    };

    const getScoreLabel = (score) => {
        if (score > 0.2) return 'Positive 🚀';
        if (score < -0.2) return 'Negative 📉';
        return 'Neutral 😐';
    };

    if (loading && !sentiment) {
        return (
            <div className="card h-full flex items-center justify-center" style={{ minHeight: '200px' }}>
                <div className="spinner"></div>
            </div>
        );
    }

    if (!sentiment) {
        return (
            <div className="card">
                <div className="flex justify-between items-center mb-md">
                    <h3 className="card-title">News Sentiment ({symbol})</h3>
                    <button className="btn btn-sm btn-outline" onClick={triggerFetch} disabled={loading}>
                        🔄 Analyze
                    </button>
                </div>
                <div className="text-center text-muted py-lg">
                    <p>No sentiment data available.</p>
                    <p className="text-sm">Click Analyze to fetch news.</p>
                </div>
            </div>
        );
    }

    const chartData = [
        { name: '1 Day', score: sentiment.sentiment_1d },
        { name: '3 Days', score: sentiment.sentiment_3d },
        { name: '7 Days', score: sentiment.sentiment_7d },
    ];

    return (
        <div className="card">
            <div className="flex justify-between items-center mb-md">
                <div>
                    <h3 className="card-title">News Sentiment ({symbol})</h3>
                    <p className="text-xs text-muted">Based on {sentiment.news_count} articles</p>
                </div>
                <div className="flex gap-sm">
                    <button className="btn btn-sm btn-ghost" onClick={fetchSentiment} title="Refresh Data">
                        🔄
                    </button>
                    <button className="btn btn-sm btn-outline" onClick={triggerFetch} disabled={loading} title="Trigger New Analysis">
                        📡 Fetch New
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-lg">
                {/* Summary Score */}
                <div className="flex flex-col justify-center items-center text-center p-md"
                    style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-md)' }}>
                    <div className="text-sm text-muted mb-xs">7-Day Trend</div>
                    <div className="text-3xl font-bold mb-xs" style={{ color: getScoreColor(sentiment.sentiment_7d) }}>
                        {sentiment.sentiment_7d > 0 ? '+' : ''}{sentiment.sentiment_7d.toFixed(2)}
                    </div>
                    <div className="badge" style={{
                        background: `${getScoreColor(sentiment.sentiment_7d)}20`,
                        color: getScoreColor(sentiment.sentiment_7d),
                        border: `1px solid ${getScoreColor(sentiment.sentiment_7d)}40`
                    }}>
                        {getScoreLabel(sentiment.sentiment_7d)}
                    </div>
                </div>

                {/* Chart */}
                <div style={{ height: '150px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData}>
                            <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
                            <YAxis domain={[-1, 1]} hide />
                            <Tooltip
                                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                contentStyle={{ background: 'var(--bg-tertiary)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px' }}
                            />
                            <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={getScoreColor(entry.score)} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="text-xs text-muted mt-md text-right">
                Last updated: {new Date(sentiment.ts_utc).toLocaleString()}
            </div>
        </div>
    );
}
