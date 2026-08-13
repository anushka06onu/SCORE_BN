import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, ShieldCheck, Database, Layers } from 'lucide-react';
import styles from './Home.module.css';

const Home = () => {
  return (
    <div className={`container animate-fade-in ${styles.homeContainer}`}>
      
      {/* Hero Section */}
      <section className={styles.hero}>
        <h1 className={styles.title}>
          Robust Severity Classification for <span className="text-gradient">Bangla Healthcare Queries</span>
        </h1>
        <p className={styles.description}>
          SCORE-BN is an experimental natural language processing system that evaluates the urgency of healthcare queries written in native Bangla, Romanized Bangla, and code-mixed formats.
        </p>
        <div className={styles.ctaGroup}>
          <Link to="/classifier" className={`btn-primary ${styles.ctaButton}`}>
            Try the Classifier <Activity size={18} />
          </Link>
          <Link to="/about" className={`btn-secondary ${styles.ctaButton}`}>
            Read Methodology
          </Link>
        </div>
      </section>

      {/* Features Section */}
      <section className={styles.features}>
        <div className={`glass-panel ${styles.featureCard}`}>
          <div className={styles.featureIcon}><Layers size={24} color="var(--cat-general)" /></div>
          <h3>Four-Level Severity</h3>
          <p>Classifies incoming real-world queries into General, Routine, Urgent, or Emergency ordered categories.</p>
        </div>
        
        <div className={`glass-panel ${styles.featureCard}`}>
          <div className={styles.featureIcon}><ShieldCheck size={24} color="var(--cat-urgent)" /></div>
          <h3>Cross-Script Robustness</h3>
          <p>Maintains consistent predictions even when users type Bangla words using English letters (Romanized).</p>
        </div>
        
        <div className={`glass-panel ${styles.featureCard}`}>
          <div className={styles.featureIcon}><Database size={24} color="var(--cat-routine)" /></div>
          <h3>Real-World Data</h3>
          <p>Trained and evaluated on 5,263 authentic healthcare queries sourced from public social media discussions.</p>
        </div>
      </section>

    </div>
  );
};

export default Home;
