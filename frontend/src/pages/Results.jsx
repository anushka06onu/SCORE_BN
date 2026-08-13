import React from 'react';
import styles from './Results.module.css';

const Results = () => {
  const modelResults = [
    { name: 'BanglaBERT', accuracy: '0.9323', f1: '0.9347', auc: '0.9882', isBest: true },
    { name: 'SCORE-BN', accuracy: '0.9183', f1: '0.9218', auc: '0.9872', isProposed: true },
    { name: 'Tuned Linear SVM', accuracy: '0.9170', f1: '0.9197', auc: '0.9841' },
    { name: 'CNN', accuracy: '0.9157', f1: '0.9186', auc: '0.9870' },
  ];

  return (
    <div className="container animate-fade-in" style={{ padding: '2rem 1.5rem' }}>
      <div className={`glass-panel ${styles.resultsContainer}`}>
        <h2 className={styles.title}>Model Comparison Results</h2>
        <p className={styles.description}>
          The following table presents the evaluation results on the held-out test split (15% of the total dataset). 
          These metrics evaluate the models' ability to classify health queries into the four severity categories.
        </p>
        
        <div className={styles.tableWrapper}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Model</th>
                <th className={styles.numberCol}>Accuracy</th>
                <th className={styles.numberCol}>Macro-F1</th>
                <th className={styles.numberCol}>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {modelResults.map((row, idx) => (
                <tr key={idx} className={row.isBest ? styles.bestRow : (row.isProposed ? styles.proposedRow : '')}>
                  <td>
                    {row.name}
                    {row.isBest && <span className={styles.badge}>Best Test</span>}
                    {row.isProposed && <span className={styles.badgeProposed}>Proposed</span>}
                  </td>
                  <td className={styles.numberCol}>{row.accuracy}</td>
                  <td className={styles.numberCol}>{row.f1}</td>
                  <td className={styles.numberCol}>{row.auc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className={styles.disclaimer}>
          <p>
            <strong>Note:</strong> Results are based on the project's held-out test split. 
            They demonstrate natural language processing capabilities on short-text classification and 
            do not represent clinical validation.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Results;
