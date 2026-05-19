import React from 'react';
import { NavLink } from 'react-router-dom';
import { Beaker, BookOpen, Home } from 'lucide-react';

export function Navigation() {
  return (
    <nav className="fixed md:top-6 bottom-4 md:bottom-auto left-1/2 -translate-x-1/2 w-[calc(100%-2rem)] md:w-[90%] max-w-5xl flex items-center justify-between bg-surface-deep/90 backdrop-blur-md border border-border-organic rounded-full px-4 py-2 md:py-3 z-[100] shadow-lg shadow-black/50">
      
      {/* Desktop Logo */}
      <div className="hidden md:block font-black text-xl text-primary-container pl-4">
        GIBD 2026
      </div>
      
      {/* Links */}
      <div className="flex items-center justify-around md:justify-end w-full md:w-auto gap-2 md:gap-4">
        <NavLink 
          to="/" 
          className={({ isActive }) => `flex flex-col md:flex-row items-center gap-1 md:gap-2 px-4 md:px-6 py-2 md:py-3 rounded-full text-xs md:text-sm font-bold transition-all ${isActive ? 'bg-primary-container text-on-primary-container shadow-[0_0_15px_rgba(255,85,0,0.4)]' : 'text-text-secondary hover:bg-secondary-container hover:text-text-primary'}`}
        >
          <Home className="w-5 h-5" />
          <span>Inicio</span>
        </NavLink>

        <NavLink 
          to="/laboratorio" 
          className={({ isActive }) => `flex flex-col md:flex-row items-center gap-1 md:gap-2 px-4 md:px-6 py-2 md:py-3 rounded-full text-xs md:text-sm font-bold transition-all ${isActive ? 'bg-primary-container text-on-primary-container shadow-[0_0_15px_rgba(255,85,0,0.4)]' : 'text-text-secondary hover:bg-secondary-container hover:text-text-primary'}`}
        >
          <Beaker className="w-5 h-5" />
          <span>Lab</span>
        </NavLink>
        
        <NavLink 
          to="/papers" 
          className={({ isActive }) => `flex flex-col md:flex-row items-center gap-1 md:gap-2 px-4 md:px-6 py-2 md:py-3 rounded-full text-xs md:text-sm font-bold transition-all ${isActive ? 'bg-primary-container text-on-primary-container shadow-[0_0_15px_rgba(255,85,0,0.4)]' : 'text-text-secondary hover:bg-secondary-container hover:text-text-primary'}`}
        >
          <BookOpen className="w-5 h-5" />
          <span className="hidden md:inline">Papers & Artículos</span>
          <span className="md:hidden">Papers</span>
        </NavLink>
      </div>
    </nav>
  );
}
