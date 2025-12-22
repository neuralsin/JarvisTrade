import { useState } from 'react';

export default function Help() {
    const [activeSection, setActiveSection] = useState('getting-started');

    const sections = {
        'getting-started': 'Getting Started',
        'paper-trading': 'Paper Trading',
        'models': 'Model Types',
        'trading': 'Trading Mechanics',
        'features': 'Features Guide',
        'faq': 'FAQ'
    };

    return (
        <div className="fade-in">
            <div className="mb-lg">
                <h1>📚 Help Center</h1>
                <p className="text-muted">Comprehensive guide to using JarvisTrade</p>
            </div>

            <div className="grid" style={{ gridTemplateColumns: '250px 1fr', gap: 'var(--spacing-lg)' }}>
                {/* Sidebar Navigation */}
                <div className="card" style={{ height: 'fit-content', position: 'sticky', top: 'var(--spacing-lg)' }}>
                    <h3 className="card-title mb-md">Topics</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-xs)' }}>
                        {Object.entries(sections).map(([id, title]) => (
                            <button
                                key={id}
                                className={activeSection === id ? 'btn btn-primary btn-sm' : 'btn btn-outline btn-sm'}
                                onClick={() => setActiveSection(id)}
                                style={{ justifyContent: 'flex-start' }}
                            >
                                {title}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Content */}
                <div className="card">
                    {activeSection === 'getting-started' && <GettingStarted />}
                    {activeSection === 'paper-trading' && <PaperTrading />}
                    {activeSection === 'models' && <Models />}
                    {activeSection === 'trading' && <TradingMechanics />}
                    {activeSection === 'features' && <FeaturesGuide />}
                    {activeSection === 'faq' && <FAQ />}
                </div>
            </div>
        </div>
    );
}

function GettingStarted() {
    return (
        <div>
            <h2 className="mb-md">🚀 Getting Started</h2>

            <section className="mb-lg">
                <h3>Welcome to JarvisTrade</h3>
                <p className="text-muted mb-md">
                    JarvisTrade is an AI-powered algorithmic trading platform that helps you make data-driven trading decisions using machine learning.
                </p>

                <div className="card" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                    <h4>✨ Key Features</h4>
                    <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                        <li>AI-powered trading signals using ML models (XGBoost, LSTM, Transformer)</li>
                        <li>Risk-free paper trading for strategy testing</li>
                        <li>Real-time portfolio tracking and analytics</li>
                        <li>Automated trade execution with advanced order types</li>
                        <li>Comprehensive backtesting framework</li>
                    </ul>
                </div>
            </section>

            <section className="mb-lg">
                <h3>Quick Start Guide</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
                    <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                            <div style={{ fontSize: '2rem' }}>1️⃣</div>
                            <div>
                                <h4 style={{ marginBottom: 'var(--spacing-xs)' }}>Train Your First Model</h4>
                                <p className="text-muted" style={{ marginBottom: 0 }}>
                                    Navigate to <strong>Models</strong> → Click <strong>Train New Model</strong> → Select model type and start training
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                            <div style={{ fontSize: '2rem' }}>2️⃣</div>
                            <div>
                                <h4 style={{ marginBottom: 'var(--spacing-xs)' }}>Activate Your Model</h4>
                                <p className="text-muted" style={{ marginBottom: 0 }}>
                                    Once trained, click on the model → Click <strong>Activate Model</strong> to use it for generating signals
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                            <div style={{ fontSize: '2rem' }}>3️⃣</div>
                            <div>
                                <h4 style={{ marginBottom: 'var(--spacing-xs)' }}>Start Paper Trading</h4>
                                <p className="text-muted" style={{ marginBottom: 0 }}>
                                    Go to <strong>Paper Trading</strong> → Monitor signals → Practice trading without real money
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                            <div style={{ fontSize: '2rem' }}>4️⃣</div>
                            <div>
                                <h4 style={{ marginBottom: 'var(--spacing-xs)' }}>Monitor Performance</h4>
                                <p className="text-muted" style={{ marginBottom: 0 }}>
                                    Check <strong>Portfolio</strong> and <strong>Trades</strong> to track your performance and refine your strategy
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
}

function PaperTrading() {
    return (
        <div>
            <h2 className="mb-md">📝 Paper Trading Explained</h2>

            <section className="mb-lg">
                <h3>What is Paper Trading?</h3>
                <p className="text-muted mb-md">
                    Paper trading is <strong>simulated trading</strong> that uses real market data but virtual money. It allows you to:
                </p>
                <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                    <li><strong>Test strategies risk-free</strong> without losing actual money</li>
                    <li><strong>Learn the platform</strong> and understand how trading works</li>
                    <li><strong>Validate ML models</strong> before deploying them with real capital</li>
                    <li><strong>Practice order execution</strong> and risk management</li>
                </ul>
            </section>

            <section className="mb-lg">
                <h3>How Paper Trading Works in JarvisTrade</h3>

                <div className="card" style={{ background: 'var(--bg-tertiary)', marginBottom: 'var(--spacing-md)' }}>
                    <h4>🔄 Real-Time Market Data</h4>
                    <p className="text-muted" style={{ marginBottom: 0 }}>
                        Paper trading uses <strong>actual live market prices</strong> from NSE (National Stock Exchange).
                        The data is identical to what you'd see in live trading, ensuring realistic simulations.
                    </p>
                </div>

                <div className="card" style={{ background: 'var(--bg-tertiary)', marginBottom: 'var(--spacing-md)' }}>
                    <h4>💰 Simulated Order Execution</h4>
                    <p className="text-muted" style={{ marginBottom: 0 }}>
                        When you place a trade, the system simulates order fills:
                    </p>
                    <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)', marginTop: 'var(--spacing-sm)' }}>
                        <li><strong>Entry:</strong> Orders are filled when market price reaches your entry price</li>
                        <li><strong>Exit:</strong> Stop-loss and target orders trigger when price hits those levels</li>
                        <li><strong>Slippage:</strong> Random slippage (0.05% - 0.15%) is added to simulate real market conditions</li>
                    </ul>
                </div>

                <div className="card" style={{ background: 'var(--bg-tertiary)', marginBottom: 'var(--spacing-md)' }}>
                    <h4>💸 Commission & Costs</h4>
                    <p className="text-muted">
                        Realistic trading costs are applied to every trade:
                    </p>
                    <div className="grid grid-cols-2" style={{ gap: 'var(--spacing-md)' }}>
                        <div>
                            <div className="text-sm text-muted">Flat Commission</div>
                            <div className="text-xl font-bold">₹20 per trade</div>
                        </div>
                        <div>
                            <div className="text-sm text-muted">Percentage Fee</div>
                            <div className="text-xl font-bold">0.05% of trade value</div>
                        </div>
                    </div>
                    <p className="text-muted" style={{ marginTop: 'var(--spacing-sm)', marginBottom: 0, fontSize: '0.875rem' }}>
                        <em>Commission = ₹20 + (0.05% × Trade Value)</em>
                    </p>
                </div>

                <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                    <h4>📊 P&L Calculation</h4>
                    <p className="text-muted" style={{ marginBottom: 'var(--spacing-sm)' }}>
                        Your profit and loss is calculated as:
                    </p>
                    <div className="card" style={{ background: 'rgba(0,0,0,0.3)', fontFamily: 'monospace', padding: 'var(--spacing-md)' }}>
                        <div>P&L = (Exit Price - Entry Price) × Quantity - Total Commission</div>
                        <div style={{ marginTop: 'var(--spacing-xs)', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                            Total Commission = Entry Commission + Exit Commission
                        </div>
                    </div>
                </div>
            </section>

            <section className="mb-lg">
                <h3>Paper vs Live Trading</h3>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.1)' }}>
                                <th style={{ padding: 'var(--spacing-sm)', textAlign: 'left' }}>Feature</th>
                                <th style={{ padding: 'var(--spacing-sm)', textAlign: 'left' }}>Paper Trading</th>
                                <th style={{ padding: 'var(--spacing-sm)', textAlign: 'left' }}>Live Trading</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <td style={{ padding: 'var(--spacing-sm)' }}><strong>Capital</strong></td>
                                <td style={{ padding: 'var(--spacing-sm)', color: 'var(--accent-success)' }}>Virtual (₹1,00,000 default)</td>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Real money</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <td style={{ padding: 'var(--spacing-sm)' }}><strong>Market Data</strong></td>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Real-time prices</td>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Real-time prices</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <td style={{ padding: 'var(--spacing-sm)' }}><strong>Risk</strong></td>
                                <td style={{ padding: 'var(--spacing-sm)', color: 'var(--accent-success)' }}>Zero (no real money)</td>
                                <td style={{ padding: 'var(--spacing-sm)', color: 'var(--accent-danger)' }}>High (can lose money)</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <td style={{ padding: 'var(--spacing-sm)' }}><strong>Order Execution</strong></td>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Simulated</td>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Real broker orders</td>
                            </tr>
                            <tr>
                                <td style={{ padding: 'var(--spacing-sm)' }}><strong>Best For</strong></td>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Learning, testing, validation</td>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Proven strategies only</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>

            <section>
                <div className="card" style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.3)' }}>
                    <h4>⚠️ Important Notes</h4>
                    <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)', marginBottom: 0 }}>
                        <li>Paper trading results may differ from live trading due to order book dynamics</li>
                        <li>Slippage in paper trading is randomized; actual slippage varies by market conditions</li>
                        <li>Always validate your strategy thoroughly in paper trading before going live</li>
                        <li>Emotional factors in live trading cannot be replicated in paper trading</li>
                    </ul>
                </div>
            </section>
        </div>
    );
}

function Models() {
    return (
        <div>
            <h2 className="mb-md">🤖 Model Types</h2>

            <p className="text-muted mb-lg">
                JarvisTrade supports three types of machine learning models, each with unique strengths for different market conditions.
            </p>

            <section className="mb-lg">
                <div className="card" style={{ background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(139, 92, 246, 0.05))' }}>
                    <h3>⚡ XGBoost (Gradient Boosting)</h3>
                    <p className="text-muted">
                        Fast, accurate tree-based ensemble model. Best for tabular feature data.
                    </p>

                    <div className="grid grid-cols-2" style={{ gap: 'var(--spacing-md)', marginTop: 'var(--spacing-md)' }}>
                        <div>
                            <h4 style={{ fontSize: '0.875rem', color: 'var(--accent-success)' }}>✅ Strengths</h4>
                            <ul style={{ marginLeft: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                <li>Fast training (minutes)</li>
                                <li>High accuracy on indicators</li>
                                <li>Handles missing data well</li>
                                <li>Great feature importance insights</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style={{ fontSize: '0.875rem', color: 'var(--accent-danger)' }}>❌ Weaknesses</h4>
                            <ul style={{ marginLeft: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                <li>Doesn't capture time sequences</li>
                                <li>Less effective for complex patterns</li>
                            </ul>
                        </div>
                    </div>

                    <div className="card" style={{ background: 'rgba(0,0,0,0.3)', marginTop: 'var(--spacing-md)' }}>
                        <strong>Best For:</strong> Quick iterations, interpretable models, feature-rich datasets
                    </div>
                </div>
            </section>

            <section className="mb-lg">
                <div className="card" style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.05))' }}>
                    <h3>🧠 LSTM (Long Short-Term Memory)</h3>
                    <p className="text-muted">
                        Recurrent neural network specialized in learning sequential patterns over time.
                    </p>

                    <div className="grid grid-cols-2" style={{ gap: 'var(--spacing-md)', marginTop: 'var(--spacing-md)' }}>
                        <div>
                            <h4 style={{ fontSize: '0.875rem', color: 'var(--accent-success)' }}>✅ Strengths</h4>
                            <ul style={{ marginLeft: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                <li>Captures time dependencies</li>
                                <li>Learns price momentum patterns</li>
                                <li>Good for trend prediction</li>
                                <li>Handles sequential data naturally</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style={{ fontSize: '0.875rem', color: 'var(--accent-danger)' }}>❌ Weaknesses</h4>
                            <ul style={{ marginLeft: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                <li>Slower training (30-60 min)</li>
                                <li>Requires more data</li>
                                <li>Less interpretable</li>
                            </ul>
                        </div>
                    </div>

                    <div className="card" style={{ background: 'rgba(0,0,0,0.3)', marginTop: 'var(--spacing-md)' }}>
                        <strong>Best For:</strong> Trend following, momentum strategies, time-series forecasting
                    </div>
                </div>
            </section>

            <section className="mb-lg">
                <div className="card" style={{ background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.2), rgba(236, 72, 153, 0.05))' }}>
                    <h3>🔮 Transformer (Attention Mechanism)</h3>
                    <p className="text-muted">
                        State-of-the-art architecture with self-attention for complex pattern recognition.
                    </p>

                    <div className="grid grid-cols-2" style={{ gap: 'var(--spacing-md)', marginTop: 'var(--spacing-md)' }}>
                        <div>
                            <h4 style={{ fontSize: '0.875rem', color: 'var(--accent-success)' }}>✅ Strengths</h4>
                            <ul style={{ marginLeft: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                <li>Advanced pattern recognition</li>
                                <li>Attention to important features</li>
                                <li>Best for complex strategies</li>
                                <li>Parallel processing capability</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style={{ fontSize: '0.875rem', color: 'var(--accent-danger)' }}>❌ Weaknesses</h4>
                            <ul style={{ marginLeft: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                                <li>Longest training time</li>
                                <li>Needs large datasets</li>
                                <li>More complex to tune</li>
                            </ul>
                        </div>
                    </div>

                    <div className="card" style={{ background: 'rgba(0,0,0,0.3)', marginTop: 'var(--spacing-md)' }}>
                        <strong>Best For:</strong> Advanced strategies, large datasets, maximum performance
                    </div>
                </div>
            </section>

            <section>
                <h3 className="mb-md">📊 Model Comparison</h3>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.1)' }}>
                                <th style={{ padding: 'var(--spacing-sm)', textAlign: 'left' }}>Metric</th>
                                <th style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>XGBoost</th>
                                <th style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>LSTM</th>
                                <th style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>Transformer</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Training Time</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center', color: 'var(--accent-success)' }}>⚡ Fast (5-10min)</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>⏱️ Medium (30-60min)</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center', color: 'var(--accent-danger)' }}>🐌 Slow (1-2hrs)</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Data Required</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>Low</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>Medium</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>High</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Interpretability</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center', color: 'var(--accent-success)' }}>High</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>Low</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>Low</td>
                            </tr>
                            <tr>
                                <td style={{ padding: 'var(--spacing-sm)' }}>Recommended For</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>Beginners</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>Intermediate</td>
                                <td style={{ padding: 'var(--spacing-sm)', textAlign: 'center' }}>Advanced</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    );
}

function TradingMechanics() {
    return (
        <div>
            <h2 className="mb-md">⚙️ Trading Mechanics</h2>

            <section className="mb-lg">
                <h3>Order Types</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
                    <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <h4>Market Order</h4>
                        <p className="text-muted" style={{ marginBottom: 0 }}>
                            Executes immediately at current market price. Best for quick entry/exit when price is less critical.
                        </p>
                    </div>

                    <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <h4>Limit Order</h4>
                        <p className="text-muted" style={{ marginBottom: 0 }}>
                            Executes only at specified price or better. Good for getting better prices but may not fill.
                        </p>
                    </div>

                    <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <h4>Stop-Limit Order</h4>
                        <p className="text-muted" style={{ marginBottom: 0 }}>
                            Becomes a limit order when stop price is hit. Useful for controlled exits.
                        </p>
                    </div>

                    <div className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <h4>Bracket Order</h4>
                        <p className="text-muted" style={{ marginBottom: 0 }}>
                            Entry + Stop Loss + Target in one order. Automated risk management for every trade.
                        </p>
                    </div>
                </div>
            </section>

            <section className="mb-lg">
                <h3>Stop Loss & Target</h3>
                <div className="card" style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', marginBottom: 'var(--spacing-md)' }}>
                    <h4>🛡️ Stop Loss</h4>
                    <p className="text-muted">
                        Automatically exits the trade when price moves against you by a specified amount.
                        <strong> Essential for limiting losses.</strong>
                    </p>
                    <div className="card" style={{ background: 'rgba(0,0,0,0.3)' }}>
                        Example: If you buy at ₹100 with a 2% stop loss, the trade exits at ₹98
                    </div>
                </div>

                <div className="card" style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                    <h4>🎯 Target</h4>
                    <p className="text-muted">
                        Automatically exits when profit target is reached. Helps lock in gains.
                    </p>
                    <div className="card" style={{ background: 'rgba(0,0,0,0.3)' }}>
                        Example: If you buy at ₹100 with a 5% target, the trade exits at ₹105
                    </div>
                </div>
            </section>

            <section className="mb-lg">
                <h3>Position Sizing</h3>
                <p className="text-muted mb-md">
                    Position sizing determines how many shares to buy. JarvisTrade calculates this based on:
                </p>
                <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                    <li><strong>Risk per Trade:</strong> Typically 1-2% of total capital</li>
                    <li><strong>Stop Loss Distance:</strong> Larger stop = smaller position size</li>
                    <li><strong>Account Balance:</strong> More capital = larger positions (proportionally)</li>
                </ul>

                <div className="card" style={{ background: 'rgba(0,0,0,0.3)', fontFamily: 'monospace', marginTop: 'var(--spacing-md)' }}>
                    Position Size = (Capital × Risk%) / (Entry Price × Stop Loss%)
                </div>
            </section>

            <section>
                <h3>Risk Management Best Practices</h3>
                <div className="card" style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.3)' }}>
                    <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)', marginBottom: 0 }}>
                        <li><strong>Never risk more than 2% per trade</strong> - Protects  from large drawdowns</li>
                        <li><strong>Always use stop losses</strong> - No exceptions, even in paper trading</li>
                        <li><strong>Maintain risk-reward ratio of at least 1:2</strong> - Target should be 2x stop distance</li>
                        <li><strong>Diversify across sectors</strong> - Don't put all capital in one stock/sector</li>
                        <li><strong>Track your performance</strong> - Review trades regularly to improve</li>
                    </ul>
                </div>
            </section>
        </div>
    );
}

function FeaturesGuide() {
    return (
        <div>
            <h2 className="mb-md">🎯 Features Guide</h2>

            <section className="mb-lg">
                <h3>Dashboard</h3>
                <p className="text-muted">
                    Central hub showing your portfolio overview, recent signals, and active positions.
                </p>
                <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                    <li>Portfolio value and P&L tracking</li>
                    <li>Recent trading signals from active model</li>
                    <li>Open positions with current P&L</li>
                    <li>Quick account statistics</li>
                </ul>
            </section>

            <section className="mb-lg">
                <h3>Models</h3>
                <p className="text-muted">
                    Train, manage, and activate machine learning models.
                </p>
                <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                    <li>Train new models with custom parameters</li>
                    <li>View model performance metrics (AUC, Precision, F1)</li>
                    <li>Feature importance analysis with SHAP values</li>
                    <li>Activate models for signal generation</li>
                    <li>Model versioning and rollback</li>
                </ul>
            </section>

            <section className="mb-lg">
                <h3>Paper Trading</h3>
                <p className="text-muted">
                    Practice trading with virtual money and real market data.
                </p>
                <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                    <li>Risk-free environment for testing strategies</li>
                    <li>Realistic order execution simulation</li>
                    <li>Commission and slippage modeling</li>
                    <li>Track performance before going live</li>
                </ul>
            </section>

            <section className="mb-lg">
                <h3>Portfolio</h3>
                <p className="text-muted">
                    Comprehensive view of your holdings and performance.
                </p>
                <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                    <li>Current positions with live P&L</li>
                    <li>Historical performance charts</li>
                    <li>Sector allocation breakdown</li>
                    <li>Risk metrics and exposure analysis</li>
                </ul>
            </section>

            <section className="mb-lg">
                <h3>Trades</h3>
                <p className="text-muted">
                    Detailed trade history and analytics.
                </p>
                <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                    <li>Complete trade log with entry/exit details</li>
                    <li>Filter by date, symbol, or outcome</li>
                    <li>Win rate and average P&L statistics</li>
                    <li>Trade performance analysis</li>
                </ul>
            </section>

            <section>
                <h3>Settings</h3>
                <p className="text-muted">
                    Configure your account and trading parameters.
                </p>
                <ul style={{ marginLeft: 'var(--spacing-md)', color: 'var(--text-secondary)' }}>
                    <li>API credentials for broker integration</li>
                    <li>Risk management settings</li>
                    <li>Notification preferences</li>
                    <li>Account security options</li>
                </ul>
            </section>
        </div>
    );
}

function FAQ() {
    const faqs = [
        {
            q: "How accurate are the ML models?",
            a: "Model accuracy varies based on market conditions and training data. XGBoost typically achieves 65-75% accuracy, while LSTM and Transformer can reach 70-80% with sufficient data. Always validate in paper trading first."
        },
        {
            q: "How often should I retrain models?",
            a: "Models automatically retrain weekly. However, you can manually retrain anytime if you notice performance degradation or want to incorporate recent market data."
        },
        {
            q: "What's the difference between paper and live mode?",
            a: "Paper mode uses virtual money for risk-free practice. Live mode uses real money and executes actual trades through your broker. Always test in paper mode first."
        },
        {
            q: "Can I customize risk parameters?",
            a: "Yes! Go to Settings to configure position sizing, risk per trade, max positions, and other risk management parameters."
        },
        {
            q: "What happens if my model's accuracy drops?",
            a: "The  system monitors model performance daily. If accuracy drops significantly (model drift), you'll receive an alert to retrain the model with fresh data."
        },
        {
            q: "How are trading signals generated?",
            a: "The active ML model analyzes technical indicators, price patterns, and other features every 15 minutes. When probability exceeds the threshold, a signal is generated."
        },
        {
            q: "Can I use multiple models simultaneously?",
            a: "Currently, only one model can be active at a time. You can switch between models by activating a different one in the Models page."
        },
        {
            q: "What's the minimum capital needed?",
            a: "For paper trading: None (virtual money). For live trading: Minimum ₹10,000 recommended to properly diversify and manage risk."
        },
        {
            q: "Are  there any hidden fees?",
            a: "No hidden fees from JarvisTrade. You'll only pay your broker's standard brokerage fees when trading live."
        },
        {
            q: "How do I transition from paper to live?",
            a: "After consistent paper trading success (3+ months recommended), verify your strategy's performance, configure broker API in Settings, then switch to Live Trading mode."
        }
    ];

    return (
        <div>
            <h2 className="mb-md">❓ Frequently Asked Questions</h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
                {faqs.map((faq, idx) => (
                    <div key={idx} className="card" style={{ background: 'var(--bg-tertiary)' }}>
                        <h4 style={{ marginBottom: 'var(--spacing-sm)', color: 'var(--accent-primary)' }}>
                            Q: {faq.q}
                        </h4>
                        <p className="text-muted" style={{ marginBottom: 0 }}>
                            <strong>A:</strong> {faq.a}
                        </p>
                    </div>
                ))}
            </div>

            <div className="card" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)', marginTop: 'var(--spacing-lg)' }}>
                <h4>💬 Still have questions?</h4>
                <p className="text-muted" style={{ marginBottom: 0 }}>
                    Check the documentation at <a href="https://github.com/yourusername/jarvistrade" style={{ color: 'var(--accent-primary)' }}>GitHub</a> or
                    reach out to our community support channels.
                </p>
            </div>
        </div>
    );
}
