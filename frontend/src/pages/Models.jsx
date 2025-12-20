import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Models() {
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState(null);
    const [loading, setLoading] = useState(true);
    const [showTrainModal, setShowTrainModal] = useState(false);
    const [trainParams, setTrainParams] = useState({
        model_name: '',
        model_type: 'xgboost',
        instrument_filter: '',
        start_date: new Date(Date.now() - 730 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 2 years ago
        end_date: new Date().toISOString().split('T')[0] // today
    });

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const response = await axios.get('/api/v1/models/');
            setModels(response.data.models);
        } catch (error) {
            console.error('Failed to fetch models:', error);
        } finally {
            setLoading(false);
        }
    };

    const fetchModelDetails = async (modelId) => {
        try {
            const response = await axios.get(`/api/v1/models/${modelId}`);
            setSelectedModel(response.data);
        } catch (error) {
            console.error('Failed to fetch model details:', error);
        }
    };

    const activateModel = async (modelId) => {
        try {
            await axios.post(`/api/v1/models/${modelId}/activate`);
            fetchModels();
            alert('Model activated successfully');
        } catch (error) {
            alert('Failed to activate model');
        }
    };

    const startTraining = async () => {
        try {
            const response = await axios.post('/api/v1/models/train', trainParams);
            alert(`Training started: ${response.data.task_id}`);
            setShowTrainModal(false);
            setTrainParams({ model_name: '', instrument_filter: '' });

            // Refresh models list after a delay
            setTimeout(fetchModels, 2000);
        } catch (error) {
            alert('Failed to start training');
        }
    };

    if (loading) {
        return <div className="flex justify-center items-center" style={{ height: '400px' }}><div className="spinner"></div></div>;
    }

    return (
        <div className="fade-in">
            <div className="flex justify-between items-center mb-lg">
                <h1>ML Models</h1>
                <button className="btn btn-primary" onClick={() => setShowTrainModal(true)}>
                    ➕ Train New Model
                </button>
            </div>

            <div className="grid grid-cols-3 gap-lg">
                {/* Models List */}
                <div className="card" style={{ gridColumn: 'span 1' }}>
                    <h3 className="card-title mb-md">Available Models</h3>
                    {models.length === 0 ? (
                        <div className="text-center text-muted" style={{ padding: 'var(--spacing-xl)' }}>
                            No models trained yet
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
                            {models.map((model) => (
                                <div
                                    key={model.id}
                                    className="card"
                                    style={{
                                        padding: 'var(--spacing-md)',
                                        cursor: 'pointer',
                                        borderColor: selectedModel?.id === model.id ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)'
                                    }}
                                    onClick={() => fetchModelDetails(model.id)}
                                >
                                    <div className="flex justify-between items-center mb-sm">
                                        <span className="font-semibold">{model.name}</span>
                                        {model.is_active && <span className="badge badge-success">Active</span>}
                                    </div>
                                    <div className="text-xs text-muted">
                                        Trained: {new Date(model.trained_at).toLocaleDateString()}
                                    </div>
                                    {model.metrics && (
                                        <div className="text-sm mt-sm">
                                            AUC: {(model.metrics.auc_roc || 0).toFixed(3)}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Model Details */}
                <div style={{ gridColumn: 'span 2' }}>
                    {selectedModel ? (
                        <div>
                            <div className="card mb-md">
                                <div className="card-header">
                                    <h3 className="card-title">{selectedModel.name}</h3>
                                    {!selectedModel.is_active && (
                                        <button
                                            className="btn btn-success btn-sm"
                                            onClick={() => activateModel(selectedModel.id)}
                                        >
                                            Activate Model
                                        </button>
                                    )}
                                </div>
                                <div className="grid grid-cols-4">
                                    <div>
                                        <div className="text-muted text-sm">AUC-ROC</div>
                                        <div className="text-xl font-bold">{(selectedModel.metrics?.auc_roc || 0).toFixed(3)}</div>
                                    </div>
                                    <div>
                                        <div className="text-muted text-sm">Precision@K</div>
                                        <div className="text-xl font-bold">{(selectedModel.metrics?.precision_at_k || 0).toFixed(3)}</div>
                                    </div>
                                    <div>
                                        <div className="text-muted text-sm">F1 Score</div>
                                        <div className="text-xl font-bold">{(selectedModel.metrics?.f1 || 0).toFixed(3)}</div>
                                    </div>
                                    <div>
                                        <div className="text-muted text-sm">Type</div>
                                        <div className="text-sm font-semibold">{selectedModel.type}</div>
                                    </div>
                                </div>
                            </div>

                            {/* Feature Importance */}
                            {selectedModel.metrics?.top_features && selectedModel.metrics.top_features.length > 0 && (
                                <div className="card">
                                    <h3 className="card-title mb-md">Feature Importance (SHAP)</h3>
                                    <ResponsiveContainer width="100%" height={400}>
                                        <BarChart data={selectedModel.metrics.top_features} layout="vertical">
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                            <XAxis type="number" stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)' }} />
                                            <YAxis dataKey="feature" type="category" width={150} stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)' }} />
                                            <Tooltip
                                                contentStyle={{ background: 'var(--bg-tertiary)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 'var(--radius-md)' }}
                                            />
                                            <Bar dataKey="importance" fill="#8b5cf6" />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="card" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <div className="text-center text-muted">
                                <span style={{ fontSize: '3rem', display: 'block', marginBottom: 'var(--spacing-md)' }}>🤖</span>
                                Select a model to view details
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Train Modal */}
            {showTrainModal && (
                <div className="modal-overlay" onClick={() => setShowTrainModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <h3 className="mb-lg">Train New Model</h3>

                        <div className="form-group">
                            <label className="form-label">Model Name</label>
                            <input
                                type="text"
                                className="form-input"
                                value={trainParams.model_name}
                                onChange={(e) => setTrainParams({ ...trainParams, model_name: e.target.value })}
                                placeholder="e.g., xgb_v1"
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label">Instrument Filter (Optional)</label>
                            <input
                                type="text"
                                className="form-input"
                                value={trainParams.instrument_filter}
                                onChange={(e) => setTrainParams({ ...trainParams, instrument_filter: e.target.value })}
                                placeholder="e.g., RELIANCE (leave blank for all)"
                            />
                        </div>

                        <div className="flex gap-md" style={{ justifyContent: 'flex-end' }}>
                            <button className="btn btn-outline" onClick={() => setShowTrainModal(false)}>
                                Cancel
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={startTraining}
                                disabled={!trainParams.model_name}
                            >
                                Start Training
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
