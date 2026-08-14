import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import Header from './components/Header';
import Home from './pages/Home';
import Classifier from './pages/Classifier';
import Results from './pages/Results';
import About from './pages/About';
import { ThemeProvider } from './context/ThemeContext';
import './App.css';

function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="app-layout">
          <Header />
          
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/classifier" element={<Classifier />} />
              <Route path="/results" element={<Results />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>

          <footer className="footer-expanded">
            <div className="container footer-grid">
              <div className="footer-col">
                <h4>SCORE-BN</h4>
                <p>Bangla Healthcare Query Severity Classification. Research Demonstration Only.</p>
              </div>
              <div className="footer-col">
                <h4>Quick Links</h4>
                <ul className="footer-links">
                  <li><Link to="/">Home</Link></li>
                  <li><Link to="/classifier">Classifier</Link></li>
                  <li><Link to="/results">Model Results</Link></li>
                  <li><Link to="/about">About & FAQ</Link></li>
                </ul>
              </div>
              <div className="footer-col">
                <h4>Contact Us</h4>
                <p>For questions or research collaboration, please reach out to our team.</p>
                <a href="mailto:contact@scorebn.dev" className="footer-email">contact@scorebn.dev</a>
              </div>
            </div>
            <div className="footer-bottom">
              <p>&copy; {new Date().getFullYear()} SCORE-BN Project.</p>
              <p className="footer-credit">Made by <strong>Fateha Hossain Anushka</strong></p>
            </div>
          </footer>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
