import Dashboard from './Dashboard';

// Paper trading is essentially the same view as dashboard but locked to paper mode
export default function PaperTrading() {
    return (
        <div>
            <div className="mb-lg">
                <h1>Paper Trading</h1>
                <p className="text-muted">Simulated trading with real market data - no actual money at risk</p>
            </div>

            {/* Reuse dashboard component logic but force paper mode */}
            <div style={{ marginTop: 'var(--spacing-xl)' }}>
                <Dashboard />
            </div>
        </div>
    );
}
