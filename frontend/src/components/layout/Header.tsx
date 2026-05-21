import { useState, useEffect } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import { Microscope, BookOpen } from 'lucide-react';
import { motion } from 'motion/react';
import { playWaterDrip } from '../../utils/audio';
import { LogoGIBD } from '../ui/LogoGIBD';

export function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  const [isScrolledPast, setIsScrolledPast] = useState(false);
  const isLandingPage = location.pathname === '/';

  const NAV_ITEMS = [
    { to: "/laboratorio", icon: Microscope, label: "Laboratorio" },
    { to: "/papers", icon: BookOpen, label: "Papers" }
  ];

  useEffect(() => {
    if (!isLandingPage) {
      setIsScrolledPast(true);
      return;
    }

    const handleScroll = () => {
      // Mobile screen: always show header
      if (window.innerWidth < 768) {
        setIsScrolledPast(true);
        return;
      }

      // Check if scroll has passed the hero buttons
      const heroButtons = document.querySelector('.hero-buttons-container');
      if (heroButtons) {
        const rect = heroButtons.getBoundingClientRect();
        // rect.bottom < 80 means the buttons have scrolled above the header zone (80px)
        setIsScrolledPast(rect.bottom < 80);
      } else {
        setIsScrolledPast(window.scrollY > 320);
      }
    };

    // Run initially
    handleScroll();

    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleScroll);
    };
  }, [isLandingPage]);

  // If mobile or not landing page or scrolled past, show header. Otherwise, hide.
  const shouldShow = !isLandingPage || isScrolledPast;

  return (
    <header className={`fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 md:px-8 py-2 md:py-2.5 bg-surface-deep/80 backdrop-blur-md rounded-full mt-4 mx-auto w-[95%] max-w-4xl border border-border-organic transition-all duration-300 ease-in-out shadow-lg shadow-black/50 ${
      shouldShow ? 'translate-y-0 opacity-100' : 'pointer-events-none -translate-y-24 opacity-0'
    }`}>
      <div className="flex items-center gap-2">
        <button 
          onClick={() => {
            navigate('/');
            playWaterDrip();
          }}
          className="flex items-center gap-2 hover:scale-105 transition-transform duration-200"
          style={{
            transform: 'translate(-19px, 0px)'
          }}
        >
          <LogoGIBD 
            className="text-primary-container w-auto" 
            style={{ height: '56px' }}
          />
        </button>
      </div>
      
      <nav className="hidden md:flex items-center p-1.5 bg-[#0F0816] rounded-full border border-border-organic relative gap-1">
        {NAV_ITEMS.map((item) => (
          <NavLink 
            key={item.to}
            to={item.to}
            onClick={playWaterDrip}
            className="relative w-40 py-2.5 flex items-center justify-center gap-2 rounded-full font-black text-xs uppercase tracking-widest transition-all duration-300 select-none z-10"
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.div
                    layoutId="header-active-pill"
                    className="absolute inset-0 pill-switch-physical rounded-full z-0 pointer-events-none"
                    transition={{ type: "spring", stiffness: 300, damping: 25 }}
                  />
                )}
                
                <item.icon className={`w-4 h-4 relative z-10 transition-all duration-300 ${
                  isActive ? 'glow-icon-orange scale-110' : 'inactive-nav-item'
                }`} />
                
                <span className={`relative z-10 transition-all duration-300 ${
                  isActive ? 'glow-orange scale-105' : 'inactive-nav-item'
                }`}>
                  {item.label}
                </span>
              </>
            )}
          </NavLink>
        ))}
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


