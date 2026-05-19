import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Microscope } from 'lucide-react';

export function Header() {
  const navigate = useNavigate();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 md:px-8 py-3 bg-surface-deep/80 backdrop-blur-md rounded-full mt-4 mx-auto w-[95%] max-w-7xl border border-border-organic transition-all duration-200 ease-in-out shadow-lg shadow-black/50">
      <div className="flex items-center gap-2">
        <button 
          onClick={() => navigate('/')}
          className="flex items-center gap-2 hover:scale-105 transition-transform duration-200"
        >
          <Microscope className="text-primary-container w-6 h-6 md:w-8 md:h-8" />
          <span className="font-bold text-xl md:text-2xl text-primary-container uppercase tracking-tighter">GIBD</span>
        </button>
      </div>
      
      <nav className="hidden md:flex items-center gap-8">
        <NavLink 
          to="/"
          className={({ isActive }) => `font-semibold text-sm hover:scale-105 transition-transform duration-200 ${isActive ? 'text-primary-container border-b-2 border-primary-container' : 'text-text-secondary hover:text-text-primary'}`}
        >
          Inicio
        </NavLink>
        <NavLink 
          to="/laboratorio"
          className={({ isActive }) => `font-semibold text-sm hover:scale-105 transition-transform duration-200 ${isActive ? 'text-primary-container border-b-2 border-primary-container' : 'text-text-secondary hover:text-text-primary'}`}
        >
          Laboratorio
        </NavLink>
        <NavLink 
          to="/papers"
          className={({ isActive }) => `font-semibold text-sm hover:scale-105 transition-transform duration-200 ${isActive ? 'text-primary-container border-b-2 border-primary-container' : 'text-text-secondary hover:text-text-primary'}`}
        >
          Papers
        </NavLink>
      </nav>

      <div className="flex items-center gap-4">
        <button className="hidden md:block bg-primary-container text-on-primary-container px-6 py-2 rounded-full font-semibold text-sm hover:scale-105 transition-transform duration-200 ripple">
          Consulta
        </button>
        <button className="md:hidden bg-primary-container text-on-primary-container p-2 rounded-full ripple">
          <Microscope className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}
