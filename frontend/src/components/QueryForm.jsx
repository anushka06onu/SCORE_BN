import React, { useState } from 'react';
import { Send, Eraser, Lightbulb } from 'lucide-react';
import styles from './QueryForm.module.css';

const exampleQueries = [
  "আমার বুকে হঠাৎ তীব্র ব্যথা হচ্ছে, দম বন্ধ হয়ে আসছে।",
  "গত তিনদিন ধরে হালকা জ্বর এবং মাথা ব্যথা।",
  "ডাক্তারবাবু, আমার বাচ্চার বয়স ৫ মাস, ওর খুব সর্দি হয়েছে, কী করব?"
];

const QueryForm = ({ onAnalyze, isLoading }) => {
  const [query, setQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState('Linear SVM');
  const [isExampleOpen, setIsExampleOpen] = useState(false);
  
  const MAX_LENGTH = 1000;
  
  const handleClear = () => {
    setQuery('');
  };
  
  const handleExampleClick = (example) => {
    setQuery(example);
    setIsExampleOpen(false);
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim().length >= 3 && !isLoading) {
      onAnalyze(query, selectedModel);
    }
  };

  const isSubmitDisabled = query.trim().length < 3 || isLoading;

  return (
    <div className={`glass-panel ${styles.formContainer}`} style={{ position: 'relative', zIndex: 50 }}>
      <form onSubmit={handleSubmit}>
        
        {/* Header & Model Selector */}
        <div className={styles.formHeader}>
          <h2 className={styles.title}>Bangla Healthcare Query</h2>
          
          <div className={styles.modelSelector}>
            <label htmlFor="model-select">Model:</label>
            <select 
              id="model-select" 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              className={styles.modelSelect}
              disabled={isLoading}
            >
              <option value="Linear SVM">Linear SVM (Fast & Accurate)</option>
              <option value="XGBoost">XGBoost</option>
              <option value="Random Forest">Random Forest</option>
              <option value="Multinomial NB">Multinomial Naive Bayes</option>
              <option value="Tuned Linear SVM">Tuned Linear SVM</option>
              <option value="Tuned XGBoost">Tuned XGBoost</option>
              <option value="Logistic Regression">Logistic Regression</option>
            </select>
          </div>
        </div>

        {/* Text Area */}
        <div className={styles.inputWrapper}>
          <textarea
            className={styles.textarea}
            placeholder="আপনার স্বাস্থ্য-সম্পর্কিত প্রশ্নটি বাংলায় লিখুন..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            maxLength={MAX_LENGTH}
            disabled={isLoading}
            rows={5}
          />
          <div className={styles.charCount}>
            {query.length} / {MAX_LENGTH}
          </div>
        </div>

        {/* Action Buttons */}
        <div className={styles.actions}>
          <div className={styles.exampleDropdown}>
            <button 
              type="button" 
              className={`btn-secondary ${styles.iconBtn}`} 
              title="Example Queries"
              onClick={() => setIsExampleOpen(!isExampleOpen)}
            >
              <Lightbulb size={18} />
              <span>Examples</span>
            </button>
            <div className={`${styles.dropdownContent} ${isExampleOpen ? styles.show : ''}`}>
              {exampleQueries.map((ex, idx) => (
                <div 
                  key={idx} 
                  className={styles.dropdownItem} 
                  onMouseDown={(e) => {
                    e.preventDefault();
                    handleExampleClick(ex);
                  }}
                >
                  {ex}
                </div>
              ))}
            </div>
          </div>

          <div className={styles.primaryActions}>
            <button type="button" className="btn-secondary" onClick={handleClear} disabled={isLoading || query.length === 0}>
              <Eraser size={18} style={{ marginRight: '0.5rem', verticalAlign: 'text-bottom' }} />
              Clear
            </button>
            
            <button type="submit" className="btn-primary" disabled={isSubmitDisabled}>
              {isLoading ? (
                <>
                  <span className={styles.spinner}></span>
                  Analyzing...
                </>
              ) : (
                <>
                  Analyze Query
                  <Send size={18} style={{ marginLeft: '0.5rem', verticalAlign: 'text-bottom' }} />
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default QueryForm;
