import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { BookOpen, Microscope } from 'lucide-react';
import { motion } from 'motion/react';
import { playWaterDrip } from '../../utils/audio';

export function BottomNav() {
  const location = useLocation();
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const scrollHeight = document.documentElement.scrollHeight;
      const clientHeight = window.innerHeight;
      
      const maxScroll = scrollHeight - clientHeight;
      const isScrollable = maxScroll > 100;
      
      // Si la página es muy larga, ocultamos cuando empieza el footer (a 750px del fondo)
      // Si es más corta, ocultamos al llegar al 85% del scroll máximo disponible
      const isNearBottom = isScrollable && (
        maxScroll > 750
          ? (clientHeight + scrollY >= scrollHeight - 750)
          : (scrollY >= maxScroll * 0.85)
      );

      if (isNearBottom) {
        setIsVisible(false);
      } else {
        // Solo en la landing page ("/") ocultamos si el scroll es menor a 50px
        if (location.pathname === '/') {
          setIsVisible(scrollY > 50);
        } else {
          setIsVisible(true);
        }
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
      className="md:hidden fixed bottom-6 left-1/2 -translate-x-1/2 w-[90%] max-w-[220px] z-50 flex justify-center gap-2 items-center px-2 py-2 bg-surface-deep/80 backdrop-blur-md border border-border-organic rounded-full shadow-lg shadow-black/50 overflow-hidden"
    >
      <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent opacity-10 pointer-events-none"></div>
      
      {[
        { to: "/laboratorio", icon: Microscope, label: "Lab" },
        { to: "/papers", icon: BookOpen, label: "Papers" }
      ].map((item) => (
        <NavLink 
          key={item.to}
          to={item.to}
          onClick={playWaterDrip}
          className="relative z-10 flex flex-col items-center justify-center rounded-full w-24 py-2 transition-all duration-300 select-none"
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <motion.div
                  layoutId="bottom-nav-indicator"
                  className="absolute inset-0 pill-switch-physical rounded-full z-0 pointer-events-none"
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                />
              )}
              <motion.div
                animate={isActive ? { rotate: [0, -15, 10, -5, 0] } : { rotate: 0 }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
                className={`relative z-10 mb-1 transition-all duration-300 ${isActive ? 'glow-icon-orange' : 'inactive-nav-item'}`}
              >
                <item.icon className="w-5 h-5" />
              </motion.div>
              <span className={`text-[10px] font-black tracking-wider uppercase relative z-10 transition-all duration-300 ${isActive ? 'glow-orange' : 'inactive-nav-item'}`}>{item.label}</span>
            </>
          )}
        </NavLink>
      ))}
    </motion.nav>

  );
}
