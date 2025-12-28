import { useState, useCallback } from 'react';

export default function CSVUploader({ onUpload, symbol, onClose }) {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [validating, setValidating] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileSelect = useCallback(async (e) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile) return;

        if (!selectedFile.name.endsWith('.csv')) {
            setError('Please select a CSV file');
            return;
        }

        setFile(selectedFile);
        setValidating(true);
        setError(null);

        // Validate via API
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const token = localStorage.getItem('token');
            const response = await fetch('/api/v1/csv/validate', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
            });

            const result = await response.json();

            if (!result.valid) {
                setError(result.error);
                setPreview(null);
            } else {
                setPreview(result);
            }
        } catch (err) {
            setError('Failed to validate file: ' + err.message);
        } finally {
            setValidating(false);
        }
    }, []);

    const handleUpload = async () => {
        if (!file || !preview?.valid) return;

        setUploading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('symbol', symbol);

            const token = localStorage.getItem('token');
            const response = await fetch('/api/v1/csv/upload', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData
            });

            const result = await response.json();

            if (result.error) {
                setError(result.error);
            } else {
                onUpload(result);
            }
        } catch (err) {
            setError('Upload failed: ' + err.message);
        } finally {
            setUploading(false);
        }
    };

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) {
            handleFileSelect({ target: { files: [droppedFile] } });
        }
    }, [handleFileSelect]);

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '700px' }}>
                <h2 className="mb-md">📁 Upload Training Data (CSV)</h2>
                <p className="text-muted mb-lg">
                    Upload historical OHLCV data for <strong>{symbol}</strong>
                </p>

                {/* Drop Zone */}
                <div
                    onDrop={handleDrop}
                    onDragOver={(e) => e.preventDefault()}
                    style={{
                        border: '2px dashed rgba(139, 92, 246, 0.5)',
                        borderRadius: 'var(--radius-lg)',
                        padding: 'var(--spacing-xl)',
                        textAlign: 'center',
                        marginBottom: 'var(--spacing-lg)',
                        background: file ? 'rgba(139, 92, 246, 0.1)' : 'transparent',
                        cursor: 'pointer'
                    }}
                    onClick={() => document.getElementById('csv-file-input').click()}
                >
                    <input
                        id="csv-file-input"
                        type="file"
                        accept=".csv"
                        onChange={handleFileSelect}
                        style={{ display: 'none' }}
                    />

                    {validating ? (
                        <div>
                            <div className="text-lg mb-sm">⏳ Validating...</div>
                            <div className="text-muted">Checking CSV format</div>
                        </div>
                    ) : file ? (
                        <div>
                            <div className="text-lg mb-sm">📄 {file.name}</div>
                            <div className="text-muted">{(file.size / 1024).toFixed(1)} KB</div>
                        </div>
                    ) : (
                        <div>
                            <div style={{ fontSize: '3rem', marginBottom: 'var(--spacing-sm)' }}>📥</div>
                            <div className="text-lg mb-sm">Drop CSV file here</div>
                            <div className="text-muted">or click to browse</div>
                        </div>
                    )}
                </div>

                {/* Error Message */}
                {error && (
                    <div className="card mb-md" style={{
                        background: 'rgba(239, 68, 68, 0.1)',
                        border: '1px solid rgba(239, 68, 68, 0.3)'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                            <span>❌</span>
                            <span>{error}</span>
                        </div>
                    </div>
                )}

                {/* Preview */}
                {preview?.valid && (
                    <div className="card mb-lg" style={{ background: 'var(--bg-tertiary)' }}>
                        <h4 className="mb-sm">✅ Valid CSV</h4>
                        <div className="grid grid-cols-3" style={{ gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-md)' }}>
                            <div>
                                <div className="text-xs text-muted">Rows</div>
                                <div className="text-lg font-bold">{preview.row_count.toLocaleString()}</div>
                            </div>
                            <div>
                                <div className="text-xs text-muted">Start Date</div>
                                <div className="text-lg">{preview.date_start}</div>
                            </div>
                            <div>
                                <div className="text-xs text-muted">End Date</div>
                                <div className="text-lg">{preview.date_end}</div>
                            </div>
                        </div>

                        <div className="text-xs text-muted mb-sm">Columns: {preview.columns.join(', ')}</div>

                        {/* Preview Table */}
                        <div style={{ overflowX: 'auto' }}>
                            <table style={{ width: '100%', fontSize: '0.75rem' }}>
                                <thead>
                                    <tr>
                                        {preview.columns.slice(0, 6).map(col => (
                                            <th key={col} style={{ padding: '4px 8px', textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                                                {col}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {preview.preview?.slice(0, 3).map((row, i) => (
                                        <tr key={i}>
                                            {preview.columns.slice(0, 6).map(col => (
                                                <td key={col} style={{ padding: '4px 8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                                    {typeof row[col] === 'number' ? row[col].toFixed(2) : row[col]}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Required Format */}
                <div className="card mb-lg" style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.3)' }}>
                    <div className="text-sm">
                        <strong>Required columns:</strong> date, open, high, low, close, volume
                    </div>
                </div>

                {/* Actions */}
                <div className="flex gap-md" style={{ justifyContent: 'flex-end' }}>
                    <button className="btn btn-outline" onClick={onClose}>
                        Cancel
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={handleUpload}
                        disabled={!preview?.valid || uploading}
                    >
                        {uploading ? '⏳ Uploading...' : '🚀 Upload & Train'}
                    </button>
                </div>
            </div>
        </div>
    );
}
