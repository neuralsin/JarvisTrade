import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../App';
import axios from 'axios';

export default function Login() {
    const [isRegister, setIsRegister] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            if (isRegister) {
                // Registration
                if (password !== confirmPassword) {
                    setError('Passwords do not match');
                    setLoading(false);
                    return;
                }

                if (password.length < 6) {
                    setError('Password must be at least 6 characters');
                    setLoading(false);
                    return;
                }

                const response = await axios.post('/api/v1/auth/register', {
                    email,
                    password
                });

                // Auto-login after registration
                const { access_token } = response.data;
                localStorage.setItem('token', access_token);
                axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

                navigate('/');
            } else {
                // Login
                await login(email, password);
                navigate('/');
            }
        } catch (err) {
            setError(err.response?.data?.detail || (isRegister ? 'Registration failed' : 'Login failed'));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center" style={{ minHeight: '100vh', background: 'var(--bg-primary)' }}>
            <div className="card" style={{ maxWidth: '420px', width: '100%' }}>
                <div style={{ textAlign: 'center', marginBottom: 'var(--spacing-2xl)' }}>
                    <h1 style={{ background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        JarvisTrade
                    </h1>
                    <p className="text-muted">ML-Driven Trading Platform</p>
                </div>

                {/* Toggle Login/Register */}
                <div style={{
                    display: 'flex',
                    gap: 'var(--spacing-sm)',
                    marginBottom: 'var(--spacing-xl)',
                    background: 'var(--bg-secondary)',
                    padding: 'var(--spacing-xs)',
                    borderRadius: 'var(--radius-lg)'
                }}>
                    <button
                        type="button"
                        onClick={() => setIsRegister(false)}
                        style={{
                            flex: 1,
                            padding: 'var(--spacing-sm) var(--spacing-md)',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            background: !isRegister ? 'var(--gradient-primary)' : 'transparent',
                            color: !isRegister ? 'var(--text-primary)' : 'var(--text-muted)',
                            fontWeight: !isRegister ? '600' : '400',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease'
                        }}
                    >
                        Sign In
                    </button>
                    <button
                        type="button"
                        onClick={() => setIsRegister(true)}
                        style={{
                            flex: 1,
                            padding: 'var(--spacing-sm) var(--spacing-md)',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            background: isRegister ? 'var(--gradient-primary)' : 'transparent',
                            color: isRegister ? 'var(--text-primary)' : 'var(--text-muted)',
                            fontWeight: isRegister ? '600' : '400',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease'
                        }}
                    >
                        Create Account
                    </button>
                </div>

                <form onSubmit={handleSubmit}>
                    {error && (
                        <div className="badge badge-danger" style={{ width: '100%', marginBottom: 'var(--spacing-md)' }}>
                            {error}
                        </div>
                    )}

                    <div className="form-group">
                        <label className="form-label">Email</label>
                        <input
                            type="email"
                            className="form-input"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            placeholder="your@email.com"
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label">Password</label>
                        <input
                            type="password"
                            className="form-input"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            placeholder="••••••••"
                            minLength={isRegister ? 6 : undefined}
                        />
                        {isRegister && (
                            <small className="text-muted" style={{ fontSize: '0.75rem', marginTop: 'var(--spacing-xs)', display: 'block' }}>
                                Minimum 6 characters
                            </small>
                        )}
                    </div>

                    {isRegister && (
                        <div className="form-group">
                            <label className="form-label">Confirm Password</label>
                            <input
                                type="password"
                                className="form-input"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                required
                                placeholder="••••••••"
                            />
                        </div>
                    )}

                    <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={loading}
                        style={{ width: '100%', marginTop: 'var(--spacing-md)' }}
                    >
                        {loading ? (
                            <div className="spinner" style={{ width: '1rem', height: '1rem' }}></div>
                        ) : (
                            isRegister ? 'Create Account' : 'Sign In'
                        )}
                    </button>
                </form>

                {!isRegister && (
                    <div style={{ marginTop: 'var(--spacing-lg)', textAlign: 'center', padding: 'var(--spacing-md)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
                        <p className="text-sm text-muted" style={{ margin: 0 }}>
                            💡 <strong>First time?</strong> Click "Create Account" above to get started!
                        </p>
                    </div>
                )}

                {isRegister && (
                    <div style={{ marginTop: 'var(--spacing-lg)', textAlign: 'center' }}>
                        <p className="text-sm text-muted">
                            By creating an account, you agree to use this platform responsibly.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
