import React from 'react';
import { AlertTriangle } from 'lucide-react';
import styles from './SafetyNotice.module.css';

const SafetyNotice = () => {
  return (
    <div className={styles.noticeContainer}>
      <div className={styles.iconWrapper}>
        <AlertTriangle size={24} className={styles.icon} />
      </div>
      <div className={styles.content}>
        <h3 className={styles.title}>Research Demonstration — Not Medical Advice</h3>
        <p className={styles.text}>
          This application is a research demonstration. It does not provide medical diagnosis, treatment, or professional medical advice. If someone may be in immediate danger, contact local emergency services or a qualified healthcare professional immediately.
        </p>
      </div>
    </div>
  );
};

export default SafetyNotice;
