import { Outlet, NavLink } from 'react-router-dom';
import { useAuth } from '../App';

export default function Layout() {
    const { user, logout } = useAuth();

    const navItems = [
        { path: '/', label: 'Dashboard', icon: '📊' },
        { path: '/portfolio', label: 'Portfolio', icon: '💼' },
        { path: '/paper-trading', label: 'Paper Trading', icon: '📝' },
        { path: '/live-trading', label: 'Live Trading', icon: '⚡' },
        { path: '/models', label: 'Models', icon: '🤖' },
        { path: '/trading-controls', label: 'Trading Controls', icon: '🎮' },
        { path: '/signal-monitor', label: 'Signal Monitor', icon: '📡' },
        { path: '/trades', label: 'Trades', icon: '💹' },
        { path: '/stocks', label: 'Manage Stocks', icon: '📈' },
        { path: '/help', label: 'Help', icon: '📚' },
        { path: '/settings', label: 'Settings', icon: '⚙️' },
    ];

    return (
        <div style={{ display: 'flex', minHeight: '100vh' }}>
            {/* Sidebar */}
            <aside style={{
                width: '250px',
                background: 'var(--bg-secondary)',
                borderRight: '1px solid rgba(255, 255, 255, 0.1)',
                padding: 'var(--spacing-xl) 0',
                display: 'flex',
                flexDirection: 'column'
            }}>
                <div style={{ padding: '0 var(--spacing-lg)', marginBottom: 'var(--spacing-2xl)' }}>
                    <h2 style={{
                        background: 'var(--gradient-primary)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        fontSize: '1.5rem'
                    }}>
                        JarvisTrade
                    </h2>
                    <p className="text-sm text-muted">{user?.email}</p>
                </div>

                <nav style={{ flex: 1 }}>
                    {navItems.map((item) => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            end={item.path === '/'}
                            style={({ isActive }) => ({
                                display: 'flex',
                                alignItems: 'center',
                                gap: 'var(--spacing-md)',
                                padding: 'var(--spacing-md) var(--spacing-lg)',
                                color: isActive ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                textDecoration: 'none',
                                background: isActive ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                                borderLeft: isActive ? '3px solid var(--accent-primary)' : '3px solid transparent',
                                transition: 'all var(--transition-base)',
                                fontWeight: isActive ? 600 : 400
                            })}
                        >
                            <span style={{ fontSize: '1.25rem' }}>{item.icon}</span>
                            {item.label}
                        </NavLink>
                    ))}
                </nav>

                <div style={{ padding: '0 var(--spacing-lg)' }}>
                    <button
                        onClick={logout}
                        className="btn btn-outline"
                        style={{ width: '100%' }}
                    >
                        Sign Out
                    </button>
                </div>
            </aside>

            {/* Main content */}
            <main style={{ flex: 1, padding: 'var(--spacing-xl)', overflowY: 'auto' }}>
                <Outlet />
            </main>
        </div>
    );
}
