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
            let response;

            // Route to V2 API for dual-model architecture
            if (formData.model_type === 'v2_dual') {
                response = await axios.post('/api/v1/models/train-v2', {
                    stock_symbol: formData.instrument_filter,
                    model_name: formData.model_name,
                    interval: formData.interval,
                    start_date: formData.start_date,
                    end_date: formData.end_date
                });
            } else {
                // V1 training
                response = await axios.post('/api/v1/models/train', formData);
            }

            // Check if response contains an error (even with 200 status)
            if (response.data.error) {
                alert(`Training failed: ${response.data.error}\n${response.data.details || ''}`);
                return;
            }

            // Success - got task_id
            if (response.data.task_id) {
                setTrainingTaskId(response.data.task_id);
                setShowWizard(false);
                const modelInfo = formData.model_type === 'v2_dual'
                    ? 'V2 Dual-Model (Direction + Quality)'
                    : formData.model_type;
                alert(`Training started! Model: ${modelInfo}\nTask ID: ${response.data.task_id}`);
            } else {
                alert('Training response missing task_id. Check backend logs.');
            }
        } catch (error) {
            const errorMsg = error.response?.data?.error || error.response?.data?.detail || error.message;
            alert(`Failed to start training: ${errorMsg}`);
        }
    };

    const handleTrainingComplete = (result) => {
        alert(`Training completed! Model: ${result.model_name}`);
        setTrainingTaskId(null);
        fetchModels();
    };

    const handleTrainingError = (error) => {
        console.error('Training failed with error:', error);
        // Error is already displayed in TrainingProgress component
        setTrainingTaskId(null);
    };

    const handleDeleteModel = async (modelId, modelName) => {
        if (!window.confirm(`Are you sure you want to delete '${modelName}'? This action cannot be undone.`)) {
            return;
        }

        try {
            await axios.delete(`/api/v1/models/${modelId}`);

            // Clear selection if deleted model was selected
            if (selectedModel?.id === modelId) {
                setSelectedModel(null);
            }

            // Refresh models list
            fetchModels();
            alert('Model deleted successfully');
        } catch (err) {
            console.error('Failed to delete model:', err);
            const errorMsg = err.response?.data?.detail || 'Failed to delete model';
            alert(errorMsg);
        }
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
                                        <div>
                                            <span className="font-semibold">{model.name}</span>
                                            {model.stock_symbol && (
                                                <span className="badge badge-primary ml-sm" style={{ fontSize: '10px' }}>
                                                    {model.stock_symbol}
                                                </span>
                                            )}
                                        </div>
                                        <div style={{ display: 'flex', gap: 'var(--spacing-xs)', alignItems: 'center' }}>
                                            {model.is_active && <span className="badge badge-success">Active</span>}
                                            <button
                                                className="btn btn-sm btn-danger"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDeleteModel(model.id, model.name);
                                                }}
                                                style={{
                                                    padding: '4px 8px',
                                                    fontSize: '12px',
                                                    minWidth: 'auto'
                                                }}
                                                title="Delete model"
                                            >
                                                🗑️
                                            </button>
                                        </div>
                                    </div>
                                    <div className="text-xs text-muted" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                        <span style={{
                                            background: model.type === 'xgboost' ? '#10b981' :
                                                model.type === 'lstm' ? '#8b5cf6' :
                                                    model.type === 'transformer' ? '#f59e0b' :
                                                        model.type?.includes('v2') ? '#ef4444' : '#6b7280',
                                            color: 'white',
                                            padding: '2px 6px',
                                            borderRadius: '4px',
                                            fontSize: '10px',
                                            textTransform: 'uppercase'
                                        }}>
                                            {model.type?.includes('v2') ? '🎯 V2' : model.type || 'xgboost'}
                                        </span>
                                        <span>Trained: {new Date(model.trained_at).toLocaleDateString()}</span>
                                    </div>
                                    {model.metrics && (
                                        <div className="text-sm mt-sm">
                                            AUC: {(model.metrics.auc_roc || model.metrics.test_auc || 0).toFixed(3)}
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
                                        <div className="text-muted text-sm">AUC</div>
                                        <div className="text-xl font-bold">
                                            {(selectedModel.metrics?.auc_roc || selectedModel.metrics?.test_auc || 0).toFixed(3)}
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-muted text-sm">
                                            {selectedModel.metrics?.precision_at_k ? 'Precision@K' : 'Test Precision'}
                                        </div>
                                        <div className="text-xl font-bold">
                                            {(selectedModel.metrics?.precision_at_k || selectedModel.metrics?.test_precision || 0).toFixed(3)}
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-muted text-sm">
                                            {selectedModel.metrics?.f1 ? 'F1 Score' : 'Test Accuracy'}
                                        </div>
                                        <div className="text-xl font-bold">
                                            {(selectedModel.metrics?.f1 || selectedModel.metrics?.test_accuracy || 0).toFixed(3)}
                                        </div>
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

