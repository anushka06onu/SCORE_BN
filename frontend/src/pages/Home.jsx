import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, ShieldCheck, Database, Layers, BarChart3, Users, Target, ChevronDown } from 'lucide-react';
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

      {/* Dataset Insights Section */}
      <section className={styles.insightsSection}>
        <h2 className={styles.sectionTitle}>Dataset Insights</h2>
        <div className={styles.insightsGrid}>
          <div className={`glass-panel ${styles.insightCard}`}>
            <Database size={32} color="var(--accent-primary)" className={styles.insightIcon} />
            <h3 className={styles.insightStat}>5,263</h3>
            <p>Total Healthcare Queries</p>
          </div>
          <div className={`glass-panel ${styles.insightCard}`}>
            <Layers size={32} color="var(--cat-urgent)" className={styles.insightIcon} />
            <h3 className={styles.insightStat}>4</h3>
            <p>Severity Classes</p>
          </div>
          <div className={`glass-panel ${styles.insightCard}`}>
            <Target size={32} color="var(--cat-routine)" className={styles.insightIcon} />
            <h3 className={styles.insightStat}>93.47%</h3>
            <p>Best Macro-F1 (BanglaBERT)</p>
          </div>
          <div className={`glass-panel ${styles.insightCard}`}>
            <Users size={32} color="var(--cat-emergency)" className={styles.insightIcon} />
            <h3 className={styles.insightStat}>3</h3>
            <p>Supported Scripts (BN, EN, Romanized)</p>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className={styles.faqSection}>
        <h2 className={styles.sectionTitle}>Frequently Asked Questions</h2>
        <div className={styles.faqGrid}>
          <div className={`glass-panel ${styles.faqCard}`}>
            <h4>What is SCORE-BN?</h4>
            <p>SCORE-BN is an experimental natural language processing model designed to classify Bangla healthcare queries into four severity levels: General, Routine, Urgent, and Emergency.</p>
          </div>
          <div className={`glass-panel ${styles.faqCard}`}>
            <h4>Is this a medical diagnosis tool?</h4>
            <p>No. This is strictly a research demonstration of NLP capabilities. It must not be used for actual medical triaging, diagnosis, or treatment.</p>
          </div>
          <div className={`glass-panel ${styles.faqCard}`}>
            <h4>What models are supported?</h4>
            <p>The system supports a variety of models including Transformers (BanglaBERT, SCORE-BN) and classical machine learning models (SVM, XGBoost, Random Forest, etc.).</p>
          </div>
          <div className={`glass-panel ${styles.faqCard}`}>
            <h4>How does the Explainable AI work?</h4>
            <p>We use LIME (Local Interpretable Model-Agnostic Explanations) to highlight the specific words in a query that most influenced the model's severity prediction.</p>
          </div>
        </div>
      </section>

    </div>
  );
};

export default Home;
