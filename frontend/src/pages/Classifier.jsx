import React, { useState } from 'react';
import SafetyNotice from '../components/SafetyNotice';
import QueryForm from '../components/QueryForm';
import PredictionCard from '../components/PredictionCard';
import { requestJson } from '../lib/request.mjs';

const Classifier = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const handlePrediction = async (query, model) => {
    setIsLoading(true);
    setResult(null);
    setErrorMessage('');
    
    try {
      const apiUrl = (import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')).replace(/\/$/, '');
      const data = await requestJson(`${apiUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: query, model }),
      });
      // Attach the original query so the PredictionCard can use it for LIME explanation
      data.queryText = query;
      setResult(data);
    } catch (error) {
      console.error(error);
      setErrorMessage(error.message || 'The backend is unavailable. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container animate-fade-in" style={{ padding: '2rem 1.5rem' }}>
      <SafetyNotice />
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        <QueryForm onAnalyze={handlePrediction} isLoading={isLoading} />
        
        {errorMessage && <p role="alert">{errorMessage}</p>}
        {result && <PredictionCard result={result} />}
      </div>
    </div>
  );
};

export default Classifier;
