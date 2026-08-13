import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/Header';
import Home from './pages/Home';
import Classifier from './pages/Classifier';
import Results from './pages/Results';
import About from './pages/About';
import './App.css'; // Just empty or globals if needed

function App() {
  return (
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

        <footer className="footer">
          &copy; {new Date().getFullYear()} SCORE-BN Project. Research Demonstration Only.
        </footer>
      </div>
    </Router>
  );
}

export default App;
