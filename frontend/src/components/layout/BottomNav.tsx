import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { BookOpen, Home, Microscope } from 'lucide-react';
import { motion } from 'motion/react';

export function BottomNav() {
  const location = useLocation();
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      // Solo en la landing page ("/") ocultamos si el scroll es menor a 50px
      if (location.pathname === '/') {
        setIsVisible(window.scrollY > 50);
      } else {
        setIsVisible(true);
      }
    };

    handleScroll();
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [location.pathname]);

  return (
    <motion.nav 
      initial={false}
      animate={{ 
        y: isVisible ? 0 : 150, 
        opacity: isVisible ? 1 : 0,
        scale: isVisible ? 1 : 0.9
      }}
      transition={{ type: "spring", stiffness: 260, damping: 25 }}
      style={{ pointerEvents: isVisible ? 'auto' : 'none' }}
      className="md:hidden fixed bottom-6 left-1/2 -translate-x-1/2 w-[90%] max-w-[380px] z-50 flex justify-around items-center px-2 py-2 bg-surface-deep/80 backdrop-blur-md border border-border-organic rounded-full shadow-lg shadow-black/50 overflow-hidden"
    >
      <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent opacity-10 pointer-events-none"></div>
      
      {[
        { to: "/", icon: Home, label: "Inicio" },
        { to: "/laboratorio", icon: Microscope, label: "Lab" },
        { to: "/papers", icon: BookOpen, label: "Papers" }
      ].map((item) => (
        <NavLink 
          key={item.to}
          to={item.to}
          className="relative z-10 flex flex-col items-center justify-center rounded-full px-5 py-2.5 text-text-secondary hover:text-white"
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <motion.div
                  layoutId="bottom-nav-indicator"
                  className="absolute inset-0 bg-gradient-to-br from-primary-container to-[#ff8c00] shadow-[0_0_15px_rgba(255,85,0,0.4)] rounded-full border border-white/20"
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                />
              )}
              <motion.div
                animate={isActive ? { rotate: [0, -15, 10, -5, 0] } : { rotate: 0 }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
                className={`relative z-10 mb-1 transition-colors duration-300 ${isActive ? 'text-white' : ''}`}
              >
                <item.icon className="w-5 h-5" />
              </motion.div>
              <span className={`text-[10px] font-black tracking-wider uppercase relative z-10 transition-colors duration-300 ${isActive ? 'text-white' : ''}`}>{item.label}</span>
            </>
          )}
        </NavLink>
      ))}
    </motion.nav>
  );
}
