import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import TrainingWizard from '../components/TrainingWizard';
import TrainingProgress from '../components/TrainingProgress';

export default function Models() {
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState(null);
    const [loading, setLoading] = useState(true);
    const [showWizard, setShowWizard] = useState(false);
    const [trainingTaskId, setTrainingTaskId] = useState(null);

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

    const handleWizardSubmit = async (formData) => {
        try {
            const response = await axios.post('/api/v1/models/train', formData);
            setTrainingTaskId(response.data.task_id);
            setShowWizard(false);
            alert(`Training started! Task ID: ${response.data.task_id}`);
        } catch (error) {
            alert(`Failed to start training: ${error.response?.data?.error || error.message}`);
        }
    };

    const handleTrainingComplete = (result) => {
        alert(`Training completed! Model: ${result.model_name}`);
        setTrainingTaskId(null);
        fetchModels();
    };

    const handleTrainingError = (error) => {
        alert('Training failed. Check console for details.');
        setTrainingTaskId(null);
    };

    if (loading) {
        return <div className="flex justify-center items-center" style={{ height: '400px' }}><div className="spinner"></div></div>;
    }

    return (
        <div className="fade-in">
            <div className="flex justify-between items-center mb-lg">
                <h1>ML Models</h1>
                <button className="btn btn-primary" onClick={() => setShowWizard(true)}>
                    ➕ Train New Model
                </button>
            </div>

            {/* Training Progress */}
            {trainingTaskId && (
                <div className="mb-lg">
                    <TrainingProgress
                        taskId={trainingTaskId}
                        onComplete={handleTrainingComplete}
                        onError={handleTrainingError}
                    />
                </div>
            )}

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

            {/* Training Wizard */}
            {showWizard && (
                <TrainingWizard
                    onSubmit={handleWizardSubmit}
                    onCancel={() => setShowWizard(false)}
                />
            )}
        </div>
    );
}
