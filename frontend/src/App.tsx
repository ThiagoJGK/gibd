import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { InteractiveBackground } from './components/ui/InteractiveBackground';
import { BottomNav } from './components/layout/BottomNav';
import { Header } from './components/layout/Header';
import { Footer } from './components/layout/Footer';
import { Landing } from './pages/Landing';
import { Laboratorio } from './pages/Laboratorio';
import { Papers } from './pages/Papers';
import { AnimatePresence, motion } from 'motion/react';

function PageWrapper({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.24, ease: [0.215, 0.61, 0.355, 1] }}
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
        <Footer />
        <BottomNav />
      </div>
    </Router>
  );
}

export default App;

