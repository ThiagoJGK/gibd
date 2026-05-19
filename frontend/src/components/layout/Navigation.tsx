import React from 'react';
import { NavLink } from 'react-router-dom';
import { Settings, Beaker, FileText, BookOpen } from 'lucide-react';
import './Navigation.css';

export function Navigation() {
  return (
    <nav className="nav-container pill-shape">
      <div className="nav-logo">
        <span className="logo-text">GIBD 2026</span>
      </div>
      
      <div className="nav-links">
        <NavLink to="/" className={({ isActive }) => `nav-link pill-shape ${isActive ? 'active' : ''}`}>
          <Beaker size={20} />
          <span>Lab</span>
        </NavLink>
        <NavLink to="/articulos" className={({ isActive }) => `nav-link pill-shape ${isActive ? 'active' : ''}`}>
          <FileText size={20} />
          <span>Artículos</span>
        </NavLink>
        <NavLink to="/papers" className={({ isActive }) => `nav-link pill-shape ${isActive ? 'active' : ''}`}>
          <BookOpen size={20} />
          <span>Papers</span>
        </NavLink>
      </div>

      <div className="nav-settings">
        <button className="settings-btn pill-shape">
          <Settings size={20} />
          <span>Mantenimiento</span>
        </button>
      </div>
    </nav>
  );
}
