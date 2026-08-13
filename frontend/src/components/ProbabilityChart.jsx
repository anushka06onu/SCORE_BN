import React from 'react';
import styles from './ProbabilityChart.module.css';

const ProbabilityChart = ({ probabilities }) => {
  // Mapping categories to specific CSS variables for colors
  const getColorClass = (category) => {
    switch(category) {
      case 'General Query': return styles.barGeneral;
      case 'Routine': return styles.barRoutine;
      case 'Urgent': return styles.barUrgent;
      case 'Emergency': return styles.barEmergency;
      default: return '';
    }
  };

  // Convert object to array for easier mapping
  const data = Object.entries(probabilities).map(([name, value]) => ({
    name,
    value: value * 100
  }));

  // Ensure standard ordering if possible
  const order = ['General Query', 'Routine', 'Urgent', 'Emergency'];
  data.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));

  return (
    <div className={styles.chartContainer}>
      {data.map((item, index) => (
        <div key={index} className={styles.row}>
          <div className={styles.label}>{item.name}</div>
          <div className={styles.barContainer}>
            <div 
              className={`${styles.bar} ${getColorClass(item.name)} animate-fade-in`} 
              style={{ width: `${Math.max(item.value, 1)}%`, animationDelay: `${index * 0.1}s` }}
            ></div>
          </div>
          <div className={styles.value}>{item.value.toFixed(1)}%</div>
        </div>
      ))}
    </div>
  );
};

export default ProbabilityChart;
