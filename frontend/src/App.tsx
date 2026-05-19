import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Navigation } from './components/layout/Navigation';
import { Landing } from './pages/Landing';
import { Laboratorio } from './pages/Laboratorio';
import { Articulos } from './pages/Articulos';
import { Papers } from './pages/Papers';

function App() {
  return (
    <Router>
      <div className="app-layout">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/laboratorio" element={<Laboratorio />} />
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
