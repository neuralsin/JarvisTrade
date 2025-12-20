import { useState, useEffect } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4'];

export default function Portfolio() {
    const [buckets, setBuckets] = useState([]);
    const [allocation, setAllocation] = useState(null);
    const [rebalance, setRebalance] = useState(null);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [newBucket, setNewBucket] = useState({
        name: '',
        bucket_type: 'sector',
        target_allocation: 0,
        rules: {}
    });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            const [bucketsRes, allocationRes, rebalanceRes] = await Promise.all([
                axios.get('/api/v1/buckets/'),
                axios.get('/api/v1/buckets/allocation'),
                axios.get('/api/v1/buckets/rebalance')
            ]);

            setBuckets(bucketsRes.data.buckets);
            setAllocation(allocationRes.data);
            setRebalance(rebalanceRes.data);
        } catch (error) {
            console.error('Failed to fetch portfolio data:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleCreateBucket = async () => {
        try {
            await axios.post('/api/v1/buckets/', newBucket);
            setShowCreateModal(false);
            setNewBucket({ name: '', bucket_type: 'sector', target_allocation: 0, rules: {} });
            fetchData();
        } catch (error) {
            alert('Failed to create bucket');
        }
    };

    const handleDeleteBucket = async (bucketId) => {
        if (!confirm('Delete this bucket?')) return;

        try {
            await axios.delete(`/api/v1/buckets/${bucketId}`);
            fetchData();
        } catch (error) {
            alert('Failed to delete bucket');
        }
    };

    if (loading) {
        return <div className="flex justify-center items-center" style={{ height: '400px' }}><div className="spinner"></div></div>;
    }

    // Prepare pie chart data
    const pieData = buckets.map((bucket, idx) => ({
        name: bucket.name,
        value: bucket.current_value || 0.01, // Avoid zero
        target: bucket.target_allocation,
        current: bucket.current_allocation
    }));

    return (
        <div className="fade-in">
            <div className="flex justify-between items-center mb-lg">
                <h1>Portfolio Buckets</h1>
                <button className="btn btn-primary" onClick={() => setShowCreateModal(true)}>
                    ➕ Create Bucket
                </button>
            </div>

            {/* Portfolio Overview */}
            <div className="grid grid-cols-3 mb-lg">
                <div className="card stat-card">
                    <div className="stat-value">₹{(allocation?.total_value || 0).toLocaleString()}</div>
                    <div className="stat-label">Total Portfolio Value</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value">{buckets.length}</div>
                    <div className="stat-label">Active Buckets</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value" style={{ color: rebalance?.priority === 'high' ? 'var(--accent-warning)' : 'var(--accent-success)' }}>
                        {rebalance?.suggestions?.length || 0}
                    </div>
                    <div className="stat-label">Rebalance Actions Needed</div>
                </div>
            </div>

            {/* Allocation Visualization */}
            <div className="grid grid-cols-2 mb-lg">
                <div className="card">
                    <h3 className="card-title">Current Allocation</h3>
                    {buckets.length > 0 ? (
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={(entry) => `${entry.name}: ${entry.current.toFixed(1)}%`}
                                    outerRadius={100}
                                    fill="#8884d8"
                                    dataKey="value"
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="text-center text-muted" style={{ padding: 'var(--spacing-2xl)' }}>
                            No buckets created yet
                        </div>
                    )}
                </div>

                <div className="card">
                    <h3 className="card-title">Target vs Actual</h3>
                    <div className="flex-col gap-md" style={{ padding: 'var(--spacing-md)' }}>
                        {buckets.map((bucket, idx) => (
                            <div key={bucket.id} style={{ marginBottom: 'var(--spacing-md)' }}>
                                <div className="flex justify-between mb-sm">
                                    <span className="font-semibold">{bucket.name}</span>
                                    <span className="text-sm">
                                        <span className={bucket.current_allocation < bucket.target_allocation ? 'text-warning' : 'text-success'}>
                                            {bucket.current_allocation.toFixed(1)}%
                                        </span>
                                        {' / '}
                                        <span className="text-muted">{bucket.target_allocation.toFixed(1)}%</span>
                                    </span>
                                </div>
                                <div style={{ position: 'relative', height: '8px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)' }}>
                                    <div
                                        style={{
                                            position: 'absolute',
                                            height: '100%',
                                            width: `${bucket.current_allocation}%`,
                                            background: COLORS[idx % COLORS.length],
                                            borderRadius: 'var(--radius-sm)',
                                            transition: 'width 0.3s'
                                        }}
                                    />
                                    {/* Target marker */}
                                    <div
                                        style={{
                                            position: 'absolute',
                                            left: `${bucket.target_allocation}%`,
                                            top: 0,
                                            height: '100%',
                                            width: '2px',
                                            background: 'white',
                                            opacity: 0.8
                                        }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Buckets List */}
            <div className="card mb-lg">
                <h3 className="card-title">Bucket Details</h3>
                <table className="table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Target %</th>
                            <th>Current %</th>
                            <th>Value</th>
                            <th>Positions</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {buckets.map((bucket) => (
                            <tr key={bucket.id}>
                                <td className="font-semibold">{bucket.name}</td>
                                <td>
                                    <span className="badge badge-info">{bucket.bucket_type}</span>
                                </td>
                                <td>{bucket.target_allocation.toFixed(1)}%</td>
                                <td>
                                    <span className={bucket.current_allocation < bucket.target_allocation ? 'text-warning' : ''}>
                                        {bucket.current_allocation.toFixed(1)}%
                                    </span>
                                </td>
                                <td>₹{bucket.current_value.toLocaleString()}</td>
                                <td>{bucket.positions_count}</td>
                                <td>
                                    <button
                                        className="btn btn-sm btn-danger"
                                        onClick={() => handleDeleteBucket(bucket.id)}
                                    >
                                        Delete
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Rebalance Suggestions */}
            {rebalance?.suggestions && rebalance.suggestions.length > 0 && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Rebalance Suggestions</h3>
                        <span className="badge badge-warning">Action Required</span>
                    </div>
                    <div className="flex-col gap-md">
                        {rebalance.suggestions.map((suggestion, idx) => (
                            <div key={idx} className="card" style={{ background: 'rgba(245, 158, 11, 0.05)', borderColor: 'var(--accent-warning)' }}>
                                <div className="flex justify-between items-center">
                                    <div>
                                        <div className="font-semibold">{suggestion.bucket}</div>
                                        <div className="text-sm text-muted">{suggestion.reason}</div>
                                    </div>
                                    <div className="text-right">
                                        <div className="font-semibold text-warning">{suggestion.action.toUpperCase()}</div>
                                        <div className="text-sm">₹{suggestion.amount.toLocaleString()}</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Create Bucket Modal */}
            {showCreateModal && (
                <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <h3 className="mb-lg">Create New Bucket</h3>

                        <div className="form-group">
                            <label className="form-label">Bucket Name</label>
                            <input
                                type="text"
                                className="form-input"
                                value={newBucket.name}
                                onChange={(e) => setNewBucket({ ...newBucket, name: e.target.value })}
                                placeholder="e.g., Tech Stocks"
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Bucket Type</label>
                            <select
                                className="form-input"
                                value={newBucket.bucket_type}
                                onChange={(e) => setNewBucket({ ...newBucket, bucket_type: e.target.value })}
                            >
                                <option value="sector">Sector</option>
                                <option value="market_cap">Market Cap</option>
                                <option value="strategy">Strategy</option>
                                <option value="custom">Custom</option>
                            </select>
                        </div>

                        <div className="form-group">
                            <label className="form-label">Target Allocation (%)</label>
                            <input
                                type="number"
                                className="form-input"
                                value={newBucket.target_allocation}
                                onChange={(e) => setNewBucket({ ...newBucket, target_allocation: parseFloat(e.target.value) })}
                                min="0"
                                max="100"
                                step="1"
                            />
                        </div>

                        <div className="flex gap-md" style={{ justifyContent: 'flex-end' }}>
                            <button className="btn btn-outline" onClick={() => setShowCreateModal(false)}>
                                Cancel
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={handleCreateBucket}
                                disabled={!newBucket.name || newBucket.target_allocation <= 0}
                            >
                                Create Bucket
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
