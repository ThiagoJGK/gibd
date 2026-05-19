import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { InteractiveBackground } from './components/ui/InteractiveBackground';
import { Navigation } from './components/layout/Navigation';
import { Header } from './components/layout/Header';
import { Landing } from './pages/Landing';
import { Laboratorio } from './pages/Laboratorio';
import { Papers } from './pages/Papers';

function App() {
  return (
    <Router>
      <InteractiveBackground />
      
      <div className="relative z-10 flex flex-col min-h-screen">
        <Header />
        <main className="flex-1 w-full pt-8 pb-24">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/laboratorio" element={<Laboratorio />} />
            <Route path="/papers" element={<Papers />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <Navigation />
      </div>
    </Router>
  );
}

export default App;
