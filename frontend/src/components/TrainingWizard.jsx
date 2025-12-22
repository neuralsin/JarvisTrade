import { useState } from 'react';

export default function TrainingWizard({ onSubmit, onCancel }) {
    const [step, setStep] = useState(1);
    const [formData, setFormData] = useState({
        model_type: '',
        model_name: '',
        instrument_filter: '',
        start_date: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        end_date: new Date().toISOString().split('T')[0]
    });

    const modelTypes = {
        xgboost: {
            name: 'XGBoost',
            icon: '⚡',
            time: '5-10 minutes',
            level: 'Beginner',
            description: 'Fast gradient boosting model. Perfect for quick iterations and getting started.',
            features: ['Fast training', 'High accuracy', 'Feature importance', 'Easy to interpret'],
            bestFor: 'Quick testing, feature analysis, daily trading'
        },
        lstm: {
            name: 'LSTM',
            icon: '🧠',
            time: '30-60 minutes',
            level: 'Intermediate',
            description: 'Neural network for sequential patterns. Excellent for trend following strategies.',
            features: ['Time sequences', 'Momentum patterns', 'Trend prediction', 'Deep learning'],
            bestFor: 'Swing trading, trend following, momentum strategies'
        },
        transformer: {
            name: 'Transformer',
            icon: '🔮',
            time: '1-2 hours',
            level: 'Advanced',
            description: 'State-of-the-art attention mechanism. Best performance with large datasets.',
            features: ['Advanced patterns', 'Attention mechanism', 'Best accuracy', 'Complex strategies'],
            bestFor: 'Professional trading, large portfolios, maximum performance'
        }
    };

    const handleNext = () => {
        if (step === 1 && !formData.model_type) {
            alert('Please select a model type');
            return;
        }
        if (step === 2 && !formData.model_name.trim()) {
            alert('Please enter a model name');
            return;
        }
        setStep(step + 1);
    };

    const handleBack = () => setStep(step - 1);

    const handleSubmit = () => {
        onSubmit(formData);
    };

    const selectModelType = (type) => {
        setFormData({ ...formData, model_type: type });
    };

    const totalStocks = formData.instrument_filter ? 1 : 9;
    const selectedModel = modelTypes[formData.model_type];

    return (
        <div className="modal-overlay" onClick={onCancel}>
            <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '900px' }}>
                {/* Progress Indicator */}
                <div style={{ marginBottom: 'var(--spacing-lg)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--spacing-sm)' }}>
                        {[1, 2, 3].map(s => (
                            <div key={s} style={{ flex: 1, textAlign: 'center' }}>
                                <div style={{
                                    display: 'inline-block',
                                    width: '32px',
                                    height: '32px',
                                    borderRadius: '50%',
                                    background: s <= step ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)',
                                    color: s <= step ? 'white' : 'var(--text-muted)',
                                    lineHeight: '32px',
                                    fontWeight: 'bold',
                                    marginBottom: 'var(--spacing-xs)'
                                }}>
                                    {s}
                                </div>
                                <div style={{ fontSize: '0.75rem', color: s <= step ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                                    {s === 1 ? 'Choose Model' : s === 2 ? 'Configure' : 'Review'}
                                </div>
                            </div>
                        ))}
                    </div>
                    <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{
                            width: `${(step / 3) * 100}%`,
                            height: '100%',
                            background: 'var(--accent-primary)',
                            transition: 'width 0.3s ease'
                        }} />
                    </div>
                </div>

                {/* Step 1: Choose Model Type */}
                {step === 1 && (
                    <div className="fade-in">
                        <h2 className="mb-md">⚡ Choose Your Model Type</h2>
                        <p className="text-muted mb-lg">
                            Select the machine learning algorithm that best fits your trading style and experience level.
                        </p>

                        <div className="grid grid-cols-3" style={{ gap: 'var(--spacing-md)' }}>
                            {Object.entries(modelTypes).map(([type, info]) => (
                                <div
                                    key={type}
                                    className="card"
                                    onClick={() => selectModelType(type)}
                                    style={{
                                        cursor: 'pointer',
                                        borderColor: formData.model_type === type ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)',
                                        borderWidth: '2px',
                                        transition: 'all 0.2s',
                                        background: formData.model_type === type ? 'rgba(139, 92, 246, 0.1)' : 'var(--bg-secondary)'
                                    }}
                                >
                                    <div style={{ fontSize: '3rem', textAlign: 'center', marginBottom: 'var(--spacing-sm)' }}>
                                        {info.icon}
                                    </div>
                                    <h3 style={{ textAlign: 'center', marginBottom: 'var(--spacing-xs)' }}>{info.name}</h3>
                                    <div className="text-sm text-muted" style={{ textAlign: 'center', marginBottom: 'var(--spacing-md)' }}>
                                        <div>⏱️ {info.time}</div>
                                        <div style={{ color: 'var(--accent-primary)' }}>👤 {info.level}</div>
                                    </div>
                                    <p className="text-muted text-sm" style={{ marginBottom: 'var(--spacing-md)' }}>
                                        {info.description}
                                    </p>
                                    <div className="text-xs text-muted">
                                        <strong>Best for:</strong> {info.bestFor}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Step 2: Configure */}
                {step === 2 && (
                    <div className="fade-in">
                        <h2 className="mb-md">📊 Configure Training</h2>

                        <div className="grid grid-cols-2" style={{ gap: 'var(--spacing-lg)' }}>
                            <div>
                                <div className="form-group">
                                    <label className="form-label">Model Name *</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        value={formData.model_name}
                                        onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                                        placeholder="e.g., my_xgb_model_v1"
                                    />
                                    <div className="text-xs text-muted" style={{ marginTop: 'var(--spacing-xs)' }}>
                                        Choose a unique, descriptive name
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Stock Filter (Optional)</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        value={formData.instrument_filter}
                                        onChange={(e) => setFormData({ ...formData, instrument_filter: e.target.value.toUpperCase() })}
                                        placeholder="e.g., RELIANCE or leave blank for all"
                                    />
                                    <div className="text-xs text-muted" style={{ marginTop: 'var(--spacing-xs)' }}>
                                        Leave blank to train on all available stocks
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Training Period</label>
                                    <select
                                        className="form-input"
                                        onChange={(e) => {
                                            const period = e.target.value;
                                            if (!period) return;

                                            const end_date = new Date().toISOString().split('T')[0];
                                            let start_date;

                                            switch (period) {
                                                case '60d':
                                                    start_date = new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
                                                    break;
                                                case '1y':
                                                    start_date = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
                                                    break;
                                                case '2y':
                                                    start_date = new Date(Date.now() - 730 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
                                                    break;
                                                case '5y':
                                                    start_date = new Date(Date.now() - 1825 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
                                                    break;
                                            }

                                            setFormData({ ...formData, start_date, end_date });
                                        }}
                                        style={{ marginBottom: 'var(--spacing-sm)' }}
                                    >
                                        <option value="">Select preset period...</option>
                                        <option value="60d">📅 Last 60 Days (15m data) - Fastest</option>
                                        <option value="1y">📅 Last 1 Year - Recommended</option>
                                        <option value="2y">📅 Last 2 Years - More Data</option>
                                        <option value="5y">📅 Last 5 Years - Maximum History</option>
                                    </select>

                                    <div className="grid grid-cols-2" style={{ gap: 'var(--spacing-sm)' }}>
                                        <div>
                                            <label className="form-label text-xs">Start Date</label>
                                            <input
                                                type="date"
                                                className="form-input"
                                                value={formData.start_date}
                                                onChange={(e) => setFormData({ ...formData, start_date: e.target.value })}
                                            />
                                        </div>
                                        <div>
                                            <label className="form-label text-xs">End Date</label>
                                            <input
                                                type="date"
                                                className="form-input"
                                                value={formData.end_date}
                                                onChange={(e) => setFormData({ ...formData, end_date: e.target.value })}
                                            />
                                        </div>
                                    </div>
                                    <div className="text-xs text-muted" style={{ marginTop: 'var(--spacing-xs)' }}>
                                        💡 More data = Better learning but slower training
                                    </div>
                                </div>
                            </div>

                            <div>
                                <div className="card" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                                    <h4 className="mb-sm">Selected Model</h4>
                                    <div style={{ fontSize: '2rem', marginBottom: 'var(--spacing-xs)' }}>
                                        {selectedModel.icon} {selectedModel.name}
                                    </div>
                                    <div className="text-sm text-muted mb-md">{selectedModel.description}</div>
                                    <div className="text-xs">
                                        <strong>Features:</strong>
                                        <ul style={{ marginLeft: 'var(--spacing-md)', marginTop: 'var(--spacing-xs)' }}>
                                            {selectedModel.features.map((f, i) => <li key={i}>{f}</li>)}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Step 3: Review */}
                {step === 3 && (
                    <div className="fade-in">
                        <h2 className="mb-md">✅ Review & Confirm</h2>
                        <p className="text-muted mb-lg">
                            Please review your configuration before starting the training process.
                        </p>

                        <div className="card" style={{ background: 'var(--bg-tertiary)', marginBottom: 'var(--spacing-lg)' }}>
                            <div className="grid grid-cols-2" style={{ gap: 'var(--spacing-lg)' }}>
                                <div>
                                    <h4 className="mb-md">Training Configuration</h4>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
                                        <div>
                                            <div className="text-xs text-muted">Model Name</div>
                                            <div className="text-lg font-semibold">{formData.model_name}</div>
                                        </div>
                                        <div>
                                            <div className="text-xs text-muted">Model Type</div>
                                            <div className="text-lg">{selectedModel.icon} {selectedModel.name}</div>
                                        </div>
                                        <div>
                                            <div className="text-xs text-muted">Training Stocks</div>
                                            <div className="text-lg">
                                                {formData.instrument_filter || 'All available stocks'} ({totalStocks} symbol{totalStocks > 1 ? 's' : ''})
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-xs text-muted">Date Range</div>
                                            <div className="text-lg">{formData.start_date} to {formData.end_date}</div>
                                        </div>
                                    </div>
                                </div>

                                <div>
                                    <h4 className="mb-md">What to Expect</h4>
                                    <div className="card" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
                                        <div style={{ marginBottom: 'var(--spacing-sm)' }}>
                                            <div className="text-xs text-muted">Estimated Training Time</div>
                                            <div className="text-xl font-bold" style={{ color: 'var(--accent-primary)' }}>
                                                ~{selectedModel.time}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-xs text-muted">Processing Steps</div>
                                            <ul className="text-sm" style={{ marginLeft: 'var(--spacing-md)', marginTop: 'var(--spacing-xs)', color: 'var(--text-muted)' }}>
                                                <li>Loading historical data</li>
                                                <li>Computing technical features</li>
                                                <li>Training {selectedModel.name} model</li>
                                                <li>Calculating performance metrics</li>
                                                <li>Saving model artifacts</li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="card" style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.3)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                                <div style={{ fontSize: '2rem' }}>💡</div>
                                <div className="text-sm">
                                    <strong>Pro Tip:</strong> You can navigate to other pages while training is in progress.
                                    We'll show you real-time progress updates.
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Navigation Buttons */}
                <div className="flex gap-md" style={{ justifyContent: 'space-between', marginTop: 'var(--spacing-xl)' }}>
                    <button className="btn btn-outline" onClick={step === 1 ? onCancel : handleBack}>
                        {step === 1 ? 'Cancel' : '❮ Back'}
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={step === 3 ? handleSubmit : handleNext}
                        disabled={step === 1 && !formData.model_type}
                    >
                        {step === 3 ? '🚀 Start Training' : 'Continue ❯'}
                    </button>
                </div>
            </div>
        </div>
    );
}
