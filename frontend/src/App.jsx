import { useState, useEffect, createContext, useContext } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import axios from 'axios';

// Context for authentication
const AuthContext = createContext(null);

export const useAuth = () => useContext(AuthContext);

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

axios.defaults.baseURL = API_BASE_URL;

// Components (will be imported from separate files)
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import PaperTrading from './pages/PaperTrading';
import LiveTrading from './pages/LiveTrading';
import Models from './pages/Models';
import Trades from './pages/Trades';
import Settings from './pages/Settings';
import Portfolio from './pages/Portfolio';
import Layout from './components/Layout';

function App() {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check if user is logged in
        const token = localStorage.getItem('token');
        if (token) {
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            fetchUser();
        } else {
            setLoading(false);
        }
    }, []);

    const fetchUser = async () => {
        try {
            const response = await axios.get('/api/v1/auth/me');
            setUser(response.data);
        } catch (error) {
            console.error('Failed to fetch user:', error);
            localStorage.removeItem('token');
            delete axios.defaults.headers.common['Authorization'];
        } finally {
            setLoading(false);
        }
    };

    const login = async (email, password) => {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        const response = await axios.post('/api/v1/auth/login', formData);
        const { access_token } = response.data;

        localStorage.setItem('token', access_token);
        axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

        await fetchUser();
        return response.data;
    };

    const register = async (email, password) => {
        const response = await axios.post('/api/v1/auth/register', {
            email,
            password
        });
        const { access_token } = response.data;

        localStorage.setItem('token', access_token);
        axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

        await fetchUser();
        return response.data;
    };

    const logout = () => {
        localStorage.removeItem('token');
        delete axios.defaults.headers.common['Authorization'];
        setUser(null);
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center" style={{ height: '100vh' }}>
                <div className="spinner"></div>
            </div>
        );
    }

    return (
        <AuthContext.Provider value={{ user, login, register, logout }}>
            <BrowserRouter>
                <Routes>
                    <Route path="/login" element={!user ? <Login /> : <Navigate to="/" />} />
                    <Route
                        path="/"
                        element={user ? <Layout /> : <Navigate to="/login" />}
                    >
                        <Route index element={<Dashboard />} />
                        <Route path="portfolio" element={<Portfolio />} />
                        <Route path="paper-trading" element={<PaperTrading />} />
                        <Route path="live-trading" element={<LiveTrading />} />
                        <Route path="models" element={<Models />} />
                        <Route path="trades" element={<Trades />} />
                        <Route path="settings" element={<Settings />} />
                    </Route>
                </Routes>
            </BrowserRouter>
        </AuthContext.Provider>
    );
}

export default App;
