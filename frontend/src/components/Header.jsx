import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Stethoscope, Activity, FileText, Info, Sun, Moon } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import styles from './Header.module.css';

const Header = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className={styles.header}>
      <div className={`container ${styles.headerContainer}`}>
        
        {/* Logo and Subtitle */}
        <Link to="/" className={styles.logoSection}>
          <div className={styles.brand}>
            <Stethoscope className={styles.brandIcon} size={28} />
            <span className={styles.brandName}>SCORE-BN</span>
          </div>
          <span className={styles.subtitle}>Bangla Healthcare Query Severity Classification</span>
        </Link>

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
          
          <button 
            onClick={toggleTheme} 
            className={styles.themeToggle}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </nav>
      </div>
    </header>
  );
};

export default Header;
