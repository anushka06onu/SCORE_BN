import React from 'react';
import { NavLink } from 'react-router-dom';
import { Stethoscope, Activity, FileText, Info } from 'lucide-react';
import styles from './Header.module.css';

const Header = () => {
  return (
    <header className={styles.header}>
      <div className={`container ${styles.headerContainer}`}>
        
        {/* Logo and Subtitle */}
        <div className={styles.logoSection}>
          <div className={styles.brand}>
            <Stethoscope className={styles.brandIcon} size={28} />
            <span className={styles.brandName}>SCORE-BN</span>
          </div>
          <span className={styles.subtitle}>Bangla Healthcare Query Severity Classification</span>
        </div>

        {/* Navigation */}
        <nav className={styles.nav}>
          <NavLink 
            to="/" 
            className={({ isActive }) => isActive ? `${styles.navLink} ${styles.active}` : styles.navLink}
          >
            Home
          </NavLink>
          
          <NavLink 
            to="/classifier" 
            className={({ isActive }) => isActive ? `${styles.navLink} ${styles.active}` : styles.navLink}
          >
            <Activity size={18} />
            Classifier
          </NavLink>
          
          <NavLink 
            to="/results" 
            className={({ isActive }) => isActive ? `${styles.navLink} ${styles.active}` : styles.navLink}
          >
            <FileText size={18} />
            Results
          </NavLink>
          
          <NavLink 
            to="/about" 
            className={({ isActive }) => isActive ? `${styles.navLink} ${styles.active}` : styles.navLink}
          >
            <Info size={18} />
            About
          </NavLink>
        </nav>
      </div>
    </header>
  );
};

export default Header;
