import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { InteractiveBackground } from './components/ui/InteractiveBackground';
import { BottomNav } from './components/layout/BottomNav';
import { Header } from './components/layout/Header';
import { Landing } from './pages/Landing';
import { Laboratorio } from './pages/Laboratorio';
import { Papers } from './pages/Papers';
import { AnimatePresence, motion } from 'motion/react';

function PageWrapper({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] } }}
      exit={{ opacity: 0, y: -10, scale: 0.99, transition: { duration: 0.15, ease: "easeIn" } }}
      className="w-full h-full"
    >
      {children}
    </motion.div>
  );
}

function AnimatedRoutes() {
  const location = useLocation();
  
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PageWrapper><Landing /></PageWrapper>} />
        <Route path="/laboratorio" element={<PageWrapper><Laboratorio /></PageWrapper>} />
        <Route path="/papers" element={<PageWrapper><Papers /></PageWrapper>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AnimatePresence>
  );
}

function App() {
  return (
    <Router>
      <InteractiveBackground />
      
      <div className="relative z-10 flex flex-col min-h-screen">
        <Header />
        <main className="flex-1 w-full pt-8 pb-24 overflow-hidden">
          <AnimatedRoutes />
        </main>
        <BottomNav />
      </div>
    </Router>
  );
}

export default App;
