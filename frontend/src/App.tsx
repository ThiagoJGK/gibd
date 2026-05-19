import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Navigation } from './components/layout/Navigation';

// Placeholder Pages
const Laboratorio = () => <div style={{ paddingTop: '100px', textAlign: 'center' }}><h1>Laboratorio GIBD</h1><p>Work in progress...</p></div>;
const Articulos = () => <div style={{ paddingTop: '100px', textAlign: 'center' }}><h1>Buscador de Artículos</h1></div>;
const Papers = () => <div style={{ paddingTop: '100px', textAlign: 'center' }}><h1>Papers e Investigaciones</h1></div>;

function App() {
  return (
    <Router>
      <div className="app-layout">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Laboratorio />} />
            <Route path="/articulos" element={<Articulos />} />
            <Route path="/papers" element={<Papers />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
