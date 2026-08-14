import React, { useState } from 'react';
import SafetyNotice from '../components/SafetyNotice';
import QueryForm from '../components/QueryForm';
import PredictionCard from '../components/PredictionCard';

const Classifier = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handlePrediction = async (query, model) => {
    setIsLoading(true);
    setResult(null);
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: query, model: model }),
      });
      
      if (!response.ok) {
        throw new Error('API Error or Model Loading');
      }
      
      const data = await response.json();
      // Attach the original query so the PredictionCard can use it for LIME explanation
      data.queryText = query;
      setResult(data);
    } catch (error) {
      console.error(error);
      alert('Error connecting to backend API. Please ensure the server is running on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container animate-fade-in" style={{ padding: '2rem 1.5rem' }}>
      <SafetyNotice />
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        <QueryForm onAnalyze={handlePrediction} isLoading={isLoading} />
        
        {result && <PredictionCard result={result} />}
      </div>
    </div>
  );
};

export default Classifier;
