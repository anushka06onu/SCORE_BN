import React from 'react';
import styles from './About.module.css';

const About = () => {
  return (
    <div className="container animate-fade-in" style={{ padding: '2rem 1.5rem' }}>
      <div className={`glass-panel ${styles.aboutContainer}`}>
        <h1 className={styles.title}>About SCORE-BN</h1>
        
        <section className={styles.section}>
          <h2>Problem Statement</h2>
          <p>
            Healthcare systems in Bangladesh face severe overcrowding and long wait times. Patients frequently lack the health literacy to determine whether their symptoms require immediate emergency care or can wait for a routine appointment. Furthermore, many online queries are written in a mix of native Bangla, Romanized Bangla (Banglish), or colloquial regional terms, making them difficult for standard models to process.
          </p>
        </section>

        <section className={styles.section}>
          <h2>The Dataset</h2>
          <p>
            The project utilizes a custom dataset of <strong>5,263 authentic healthcare queries</strong> sourced from public social media platforms and health forums.
          </p>
          <ul className={styles.list}>
            <li><strong>Total Samples:</strong> 5,263</li>
            <li><strong>Train Split:</strong> 3,650 (70%)</li>
            <li><strong>Validation Split:</strong> 782 (15%)</li>
            <li><strong>Test Split:</strong> 783 (15%)</li>
          </ul>
        </section>

        <section className={styles.section}>
          <h2>Severity Categories (Ordinal)</h2>
          <p>The queries are classified into four ordered severity levels:</p>
          <div className={styles.categories}>
            <div className={`${styles.category} ${styles.general}`}>
              <strong>General Query:</strong> General informational or administrative query.
            </div>
            <div className={`${styles.category} ${styles.routine}`}>
              <strong>Routine:</strong> Routine professional consultation may be appropriate.
            </div>
            <div className={`${styles.category} ${styles.urgent}`}>
              <strong>Urgent:</strong> Prompt professional assessment may be appropriate.
            </div>
            <div className={`${styles.category} ${styles.emergency}`}>
              <strong>Emergency:</strong> Potentially time-sensitive; seek immediate professional assistance.
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <h2>Methodology</h2>
          <p>
            The <strong>SCORE-BN</strong> (Severity Classification and Ordinal Regression for English-Bangla) model is the proposed methodology. It introduces a specialized ordinal regression penalty to heavily penalize dangerous under-predictions (e.g., predicting an Emergency as a General Query). 
          </p>
          <p>
            However, standard <strong>BanglaBERT</strong> achieved the highest absolute performance across standard metrics (Macro-F1 and Accuracy) and serves as the primary engine for this demonstration.
          </p>
        </section>

        <section className={styles.section}>
          <h2>Ethical Limitations</h2>
          <div className={styles.warningBox}>
            <p>
              <strong>Not Medical Advice:</strong> This system does not diagnose diseases, recommend medications, or replace human medical professionals. It is purely an NLP research demonstration focusing on triage severity classification. It may produce inaccurate predictions, especially on complex or unseen colloquial text.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
};

export default About;
