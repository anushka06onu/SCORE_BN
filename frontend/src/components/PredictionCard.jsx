import React, { useState } from 'react';
import { AlertCircle, Clock, Cpu, ChevronDown, ChevronUp } from 'lucide-react';
import ProbabilityChart from './ProbabilityChart';
import styles from './PredictionCard.module.css';

const PredictionCard = ({ result }) => {
  const [showExplanation, setShowExplanation] = useState(false);
  const [explanation, setExplanation] = useState(null);
  const [loadingExplain, setLoadingExplain] = useState(false);
  
  if (!result) return null;

  const {
    predicted_class,
    confidence,
    probabilities,
    model_used,
    processing_time_ms,
    queryText
  } = result;

  const confidencePercent = (confidence * 100).toFixed(1);
  
  // Determine severity class and color mapping
  const severityClass = predicted_class.toLowerCase().replace(' ', '-');
  
  let confidenceLevel = 'High';
  let confidenceNotice = null;
  
  if (confidence < 0.50) {
    confidenceLevel = 'Low';
    confidenceNotice = "Low confidence—human review recommended";
  } else if (confidence < 0.75) {
    confidenceLevel = 'Moderate';
    confidenceNotice = "Moderate confidence";
  }

  const handleExplainToggle = async () => {
    const nextState = !showExplanation;
    setShowExplanation(nextState);

    if (nextState && !explanation) {
      setLoadingExplain(true);
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/api/explain`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: queryText, model: model_used }),
        });
        if (response.ok) {
          const data = await response.json();
          setExplanation(data.explanation);
        }
      } catch (error) {
        console.error('LIME explanation error:', error);
      } finally {
        setLoadingExplain(false);
      }
    }
  };

  const generateExplanationText = (expData, pClass) => {
    if (!expData || expData.length === 0) return null;
    
    // Sort by absolute weight
    const sorted = [...expData].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    const topPositive = sorted.filter(item => item.weight > 0).slice(0, 3);
    const topNegative = sorted.filter(item => item.weight < 0).slice(0, 2);
    
    if (topPositive.length === 0) {
      return `The model classified this as ${pClass}, but could not identify strong contributing words.`;
    }
    
    const posWordsStr = topPositive.map(w => `'${w.word}'`).join(' and ');
    let text = `The AI classified your query as **${pClass}** primarily because the presence of the word${topPositive.length > 1 ? 's' : ''} **${posWordsStr}** strongly indicates this severity level.`;
    
    if (topNegative.length > 0) {
      const negWordsStr = topNegative.map(w => `'${w.word}'`).join(' and ');
      text += ` Conversely, the word${topNegative.length > 1 ? 's' : ''} **${negWordsStr}** slightly pulled the prediction away from this category, but were outweighed by the positive indicators.`;
    }
    
    return text;
  };

  return (
    <div className={`glass-panel animate-fade-in ${styles.cardContainer}`}>
      
      {/* Top Banner (Severity) */}
      <div className={`${styles.severityBanner} ${styles[severityClass]}`}>
        <div className={styles.severityLabel}>
          <span>Predicted Severity</span>
          <h2>{predicted_class}</h2>
        </div>
        
        <div className={styles.confidenceLabel}>
          <span>Confidence</span>
          <div className={styles.confidenceValue}>
            {confidencePercent}%
          </div>
        </div>
      </div>

      {/* Warning Notice if Low Confidence */}
      {confidenceNotice && (
        <div className={styles.warningBox}>
          <AlertCircle size={16} />
          {confidenceNotice}
        </div>
      )}

      {/* Main Content Area */}
      <div className={styles.contentArea}>
        
        <h3 className={styles.chartTitle}>Class Probabilities</h3>
        <ProbabilityChart probabilities={probabilities} />
        
        <div className={styles.metaInfo}>
          <div className={styles.metaItem}>
            <Cpu size={16} />
            <span>Model used: <strong>{model_used}</strong></span>
          </div>
          <div className={styles.metaItem}>
            <Clock size={16} />
            <span>Processing time: <strong>{(processing_time_ms / 1000).toFixed(2)} seconds</strong></span>
          </div>
        </div>
        
        {/* Explanation Toggle */}
        <div className={styles.explanationSection}>
          <button 
            className={styles.explanationToggle}
            onClick={handleExplainToggle}
          >
            Why did the model make this prediction?
            {showExplanation ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
          
          {showExplanation && (
            <div className={`animate-fade-in ${styles.explanationContent}`}>
              {loadingExplain ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className={styles.spinnerDark}></span> Generating LIME explanation...
                </div>
              ) : explanation ? (
                <div>
                  <div className={styles.limeDescription} style={{ marginBottom: '1.5rem', backgroundColor: 'rgba(59, 130, 246, 0.05)', padding: '1rem', borderRadius: '0.5rem', borderLeft: '3px solid var(--accent-primary)' }}>
                    <p dangerouslySetInnerHTML={{ __html: generateExplanationText(explanation, predicted_class).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}></p>
                  </div>
                  
                  <p style={{ marginBottom: '1rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    <strong>Word contributions towards "{predicted_class}":</strong>
                  </p>
                  <div className={styles.tagCloud}>
                    {explanation.map((item, idx) => (
                      <span 
                        key={idx} 
                        className={styles.wordTag}
                        style={{ 
                          backgroundColor: item.weight > 0 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                          color: item.weight > 0 ? '#34d399' : '#fca5a5',
                          border: `1px solid ${item.weight > 0 ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`
                        }}
                      >
                        {item.word} ({(item.weight > 0 ? '+' : '') + item.weight.toFixed(3)})
                      </span>
                    ))}
                  </div>
                  <div className={styles.limeDescription}>
                    <p>
                      This section utilizes <strong>LIME (Local Interpretable Model-Agnostic Explanations)</strong> to demonstrate the transparency of the model. 
                      The highlighted words show which parts of the query most heavily influenced the prediction.
                    </p>
                    <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                      <span style={{ color: '#10b981', fontWeight: 600 }}>Green</span> words increased the probability of this class, while <span style={{ color: '#ef4444', fontWeight: 600 }}>Red</span> words decreased it.
                    </p>
                  </div>
                </div>
              ) : (
                <p>Explanation unavailable.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;
