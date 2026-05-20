import { NavLink } from 'react-router-dom';
import { Mail, MapPin, Globe, ArrowUpRight } from 'lucide-react';
import { LogoUTN } from '../ui/LogoUTN';
import { playWaterDrip } from '../../utils/audio';

export function Footer() {
  const currentYear = new Date().getFullYear();

  const handleLinkClick = () => {
    playWaterDrip();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="w-full relative overflow-hidden bg-surface-deep/70 backdrop-blur-md border-t border-border-organic/80 py-16 px-6 pb-32 md:pb-16 mt-auto">
      {/* Decorative background glow */}
      <div className="absolute -bottom-24 -left-24 w-80 h-80 bg-primary-container/5 rounded-full blur-3xl pointer-events-none"></div>
      <div className="absolute -top-24 -right-24 w-80 h-80 bg-border-organic/15 rounded-full blur-3xl pointer-events-none"></div>

      <div className="max-w-7xl mx-auto relative z-10 flex flex-col gap-12">
        {/* Main Footer Grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 lg:gap-12">
          {/* Column 1: Brand / Description */}
          <div className="md:col-span-5 flex flex-col gap-5 text-center md:text-left">
            <div className="flex justify-center md:justify-start items-center">
              <h3 
                onClick={() => {
                  playWaterDrip();
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className="text-2xl md:text-3xl font-black text-text-primary tracking-widest uppercase glow-orange hover:scale-105 transition-transform duration-300 cursor-pointer select-none"
              >
                GIBD
              </h3>
            </div>
            <p className="text-text-secondary text-sm md:text-base font-medium leading-relaxed max-w-md mx-auto md:mx-0">
              Grupo de Investigación en Big Data (GIBD). Desarrollamos soluciones avanzadas de Inteligencia Artificial, Ciencia de Datos y computación científica con impacto real, local y global.
            </p>
          </div>

          {/* Column 2: Navigation Links */}
          <div className="md:col-span-3 md:col-start-7 flex flex-col gap-4 text-center md:text-left">
            <h5 className="font-black text-xs uppercase tracking-widest text-primary-container/90">
              Navegación
            </h5>
            <ul className="flex flex-col gap-3 font-semibold text-sm">
              <li>
                <NavLink
                  to="/"
                  onClick={handleLinkClick}
                  className={({ isActive }) =>
                    `inline-flex items-center gap-1.5 transition-all duration-300 hover:text-primary-container hover:translate-x-1 ${
                      isActive ? 'text-primary-container' : 'text-text-secondary'
                    }`
                  }
                >
                  Inicio
                </NavLink>
              </li>
              <li>
                <NavLink
                  to="/laboratorio"
                  onClick={handleLinkClick}
                  className={({ isActive }) =>
                    `inline-flex items-center gap-1.5 transition-all duration-300 hover:text-primary-container hover:translate-x-1 ${
                      isActive ? 'text-primary-container' : 'text-text-secondary'
                    }`
                  }
                >
                  Laboratorio de IA
                </NavLink>
              </li>
              <li>
                <NavLink
                  to="/papers"
                  onClick={handleLinkClick}
                  className={({ isActive }) =>
                    `inline-flex items-center gap-1.5 transition-all duration-300 hover:text-primary-container hover:translate-x-1 ${
                      isActive ? 'text-primary-container' : 'text-text-secondary'
                    }`
                  }
                >
                  Papers & Artículos
                </NavLink>
              </li>
            </ul>
          </div>

          {/* Column 3: Location and Institutional info */}
          <div className="md:col-span-3 flex flex-col gap-4 text-center md:text-left">
            <h5 className="font-black text-xs uppercase tracking-widest text-primary-container/90">
              Contacto y Sede
            </h5>
            <ul className="flex flex-col gap-3.5 text-sm text-text-secondary font-medium">
              <li className="flex items-start gap-2.5 justify-center md:justify-start">
                <MapPin className="w-5 h-5 text-primary-container/70 shrink-0 mt-0.5" />
                <span className="leading-snug">
                  UTN FRCU, Ing. Pereyra 676
                  <br />
                  Concepción del Uruguay, ER, Argentina
                </span>
              </li>
              <li className="flex items-center gap-2.5 justify-center md:justify-start">
                <Mail className="w-4.5 h-4.5 text-primary-container/70 shrink-0" />
                <a 
                  href="mailto:gibd@frcu.utn.edu.ar"
                  className="hover:text-primary-container transition-colors duration-200"
                >
                  gibd@frcu.utn.edu.ar
                </a>
              </li>
              <li className="flex items-center gap-2.5 justify-center md:justify-start">
                <Globe className="w-4.5 h-4.5 text-primary-container/70 shrink-0" />
                <a
                  href="https://www.frcu.utn.edu.ar"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-primary-container transition-colors duration-200 inline-flex items-center gap-0.5"
                >
                  <span>www.frcu.utn.edu.ar</span>
                  <ArrowUpRight className="w-3.5 h-3.5 opacity-60" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Horizontal separator */}
        <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-border-organic/70 to-transparent"></div>

        {/* Bottom Bar: Copyright & Institutional Co-branding */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-6 text-center sm:text-left">
          <div className="flex flex-col gap-1.5">
            <p className="text-xs text-text-secondary font-semibold">
              © {currentYear} Grupo de Investigación en Big Data (GIBD). Todos los derechos reservados.
            </p>
            <p className="text-[10px] text-text-secondary/60 font-medium">
              Desarrollado y mantenido por investigadores de la Facultad Regional Concepción del Uruguay.
            </p>
          </div>

          <div className="flex items-center gap-3 bg-secondary-container/30 border border-border-organic/40 px-4 py-2.5 rounded-2xl backdrop-blur-sm">
            <span className="text-[11px] font-bold uppercase tracking-wider text-text-secondary/80">
              UTN FRCU
            </span>
            <span className="w-[1px] h-3.5 bg-border-organic/60"></span>
            <LogoUTN 
              className="w-7 h-7 text-text-primary hover:text-primary-container hover:scale-110 hover:filter hover:drop-shadow-[0_0_6px_rgba(255,85,0,0.6)] transition-all duration-300 cursor-pointer"
              onClick={() => {
                playWaterDrip();
                window.open('https://www.frcu.utn.edu.ar', '_blank');
              }}
            />
          </div>
        </div>
      </div>
    </footer>
  );
}
