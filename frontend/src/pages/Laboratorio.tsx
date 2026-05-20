import React, { useState, useRef, useEffect } from 'react';
import { Microscope, Search, Archive, Database, FileText, History, Network, Scale, Sparkles, TerminalSquare, Video, AudioLines } from 'lucide-react';
import { motion } from 'motion/react';

const MEDIA_CONFIG = [
  { id: 'Imagen', angle: -35, width: 110 },
  { id: 'Video', angle: -12, width: 100 },
  { id: 'Texto', angle: 12, width: 100 },
  { id: 'Sonido', angle: 35, width: 110 }
] as const;

type MediaType = typeof MEDIA_CONFIG[number]['id'];

const MODELS_BY_MEDIA: Record<MediaType, string[]> = {
  'Imagen': ['Marcas de Ganado', 'OBC (Satelital)'],
  'Video': ['Análisis de Comportamiento', 'Seguimiento de Objetos'],
  'Texto': ['PTAH-Jurídico', 'Archivos NLP'],
  'Sonido': ['Análisis de Frecuencia', 'Reconocimiento de Voz', 'Detección de Anomalías']
};

const SIDEBAR_ITEMS = [
  { id: 1, icon: FileText, label: 'Documento: 8 Articulo: 3' },
  { id: 2, icon: Scale, label: 'Norma General' },
  { id: 3, icon: Network, label: 'Análisis de Red' },
  { id: 4, icon: Database, label: 'Dataset 2024' },
  { id: 5, icon: Archive, label: 'Historial 2023', opacity: 'opacity-60' },
  { id: 6, icon: History, label: 'Archivos NLP', opacity: 'opacity-60' },
];

const REFERENCE_IMAGES = [
  "https://lh3.googleusercontent.com/aida-public/AB6AXuAAnKG6Zsl4FZOrlhof0HSaEMgu1fTJ21VcBiVxCDu1PCcs8RixieYZfQGgdIFTzxHrEnIuH9PZCAG8umu0LhTKmiY8SO1SxrmJFlOUJa_x82LPfS2lemdGMXbwvcXZnFIirjwHLoigFJnNMQAzsIxmzqqth6AYreOjnMpJneGHZVxbhnNo5ewOMSvJat2d3S7ENB7xyVWffRQqHGfBPgMZly0a6RbTLsXLCJxRH0T38dEONLu67HwGPmZB-cpVwfEbo_l2f4E0m0c",
  "https://lh3.googleusercontent.com/aida-public/AB6AXuCaNxVCUkV1zq5TjhTYv5oYOemqRnt4VcFnrcrcHVUkwYVlcZX3VxvkZ6N8f4nctOnkDMDGXHcr_ZhfOMaFKGdU2gQUUcFkKcp8Ryy8l83x_TFm8RNPoEvzgClFAS7fZ7WRi8ZJMhe0b0qt_CzTSJllgoiTDW1HxubaNPIqOi5QKeiIz6JKFbq-YZaKSjxBqPXUX9LbIUPD-cKWl146PNYKGLH2-bkdQlp2cQ5CukLyu8UblPtFHuEakkja7fxrdEIQ6MrDHILcYA8",
  "https://lh3.googleusercontent.com/aida-public/AB6AXuAiqTUmRcQ_D5YHQkGdpBN95kWJoH5zlaNTmnBm4wNBOj_MEmfoJC-7jpx-eaYeddlZNimERmQxQVFYVq8-n-xX5IEQ97YaW2l7TKkGfQ9J3y-mjI7aIZX7ZgAPYLb1d2xkEf35Gr0PUjAXs-88-wXzsJ4ySPYcUN9zv2l0Pm7p9q6_iBZkHkAltzV0VTy78FpfFgyQFQrFvRX4NvCsMsoPVqehRgZ4I2Ndld6KxhLDTdV7jMGGOoEIAT-YD_mETjPAyXHHh3fE1M8",
  "https://lh3.googleusercontent.com/aida-public/AB6AXuDIIzVhor0geVVwCFjltkZYlQfwDpiCfLGtLSA1JFTYTfTMMG-jAfahYwS_64iHbi_RBx3adFmvbh1ILBWb38_N7QFD3fa0tMTWMIqhd2EVusAWYEdNHgsowwU7Z_deEH1GGuaRzth1MAuhXtIpqh9OXpGD5Flnpc_go3dqwJfS72KkG5-N4wlyx19o5exNn3K5hzldyUdh-A2SnZ2sKwXhHX-Gr0hymyN8ItapW9b93ukKMkNcjsV-WabF0N6WcnOh2QMNDzK3USA"
];

// Cache global de AudioContext para reducir la presión del recolector de basura (GC) y consumo de RAM
let cachedAudioCtx: AudioContext | null = null;
const getAudioCtx = (): AudioContext | null => {
  if (typeof window === 'undefined') return null;
  if (!cachedAudioCtx) {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (AudioContextClass) {
      cachedAudioCtx = new AudioContextClass();
    }
  }
  // Reanudar si el contexto está suspendido (política del navegador)
  if (cachedAudioCtx && cachedAudioCtx.state === 'suspended') {
    cachedAudioCtx.resume().catch(() => {});
  }
  return cachedAudioCtx;
};

// Audio feedback físico de alta fidelidad: Clac mecánico ( Cherry MX / Rotary Switch )
const playMechanicalClick = () => {
  try {
    const audioCtx = getAudioCtx();
    if (!audioCtx) return;

    const now = audioCtx.currentTime;

    // Componente 1: Transitorio de Impacto Metálico/Plástico (Ruido Blanco Filtrado)
    const bufferSize = audioCtx.sampleRate * 0.025; // 25ms de transitorio
    const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      channelData[i] = Math.random() * 2 - 1;
    }

    const noiseSource = audioCtx.createBufferSource();
    noiseSource.buffer = buffer;

    const noiseFilter = audioCtx.createBiquadFilter();
    noiseFilter.type = 'bandpass';
    noiseFilter.frequency.setValueAtTime(3200, now);
    noiseFilter.Q.setValueAtTime(1.8, now);

    const noiseGain = audioCtx.createGain();
    noiseGain.gain.setValueAtTime(0.06, now); // Volumen optimizado y más sutil
    noiseGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.02);

    noiseSource.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(audioCtx.destination);

    // Componente 2: Cuerpo de resonancia del interruptor (Frecuencia Media-Baja)
    const bodyOsc = audioCtx.createOscillator();
    bodyOsc.type = 'triangle'; // Sonido con armónicos naturales cálidos
    bodyOsc.frequency.setValueAtTime(480, now);
    bodyOsc.frequency.exponentialRampToValueAtTime(140, now + 0.06);

    const bodyGain = audioCtx.createGain();
    bodyGain.gain.setValueAtTime(0.11, now); // Volumen optimizado y más sutil
    bodyGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.06);

    bodyOsc.connect(bodyGain);
    bodyGain.connect(audioCtx.destination);

    // Componente 3: Golpe bajo sordo (Frecuencia Baja para peso físico)
    const thudOsc = audioCtx.createOscillator();
    thudOsc.type = 'sine';
    thudOsc.frequency.setValueAtTime(110, now);
    thudOsc.frequency.exponentialRampToValueAtTime(60, now + 0.09);

    const thudGain = audioCtx.createGain();
    thudGain.gain.setValueAtTime(0.07, now); // Volumen optimizado y más sutil
    thudGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.09);

    thudOsc.connect(thudGain);
    thudGain.connect(audioCtx.destination);

    // Reproducción simultánea
    noiseSource.start(now);
    noiseSource.stop(now + 0.025);
    bodyOsc.start(now);
    bodyOsc.stop(now + 0.06);
    thudOsc.start(now);
    thudOsc.stop(now + 0.09);
  } catch (e) {
    // Ignorar si el navegador bloquea audio sin interacción del usuario
  }
};

// Micro-tick metálico/plástico para el engranaje del dial
const playGearTick = () => {
  try {
    const audioCtx = getAudioCtx();
    if (!audioCtx) return;

    const now = audioCtx.currentTime;

    // Transitorio de impacto rápido (10ms)
    const bufferSize = audioCtx.sampleRate * 0.01;
    const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      channelData[i] = Math.random() * 2 - 1;
    }

    const noiseSource = audioCtx.createBufferSource();
    noiseSource.buffer = buffer;

    const noiseFilter = audioCtx.createBiquadFilter();
    noiseFilter.type = 'highpass';
    noiseFilter.frequency.setValueAtTime(4500, now);

    const noiseGain = audioCtx.createGain();
    noiseGain.gain.setValueAtTime(0.02, now); // Volumen optimizado y más sutil
    noiseGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.008);

    noiseSource.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(audioCtx.destination);

    // Resonancia de la rueda dentada
    const clickOsc = audioCtx.createOscillator();
    clickOsc.type = 'sine';
    clickOsc.frequency.setValueAtTime(780, now);
    clickOsc.frequency.exponentialRampToValueAtTime(320, now + 0.015);

    const clickGain = audioCtx.createGain();
    clickGain.gain.setValueAtTime(0.03, now); // Volumen optimizado y más sutil
    clickGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.015);

    clickOsc.connect(clickGain);
    clickGain.connect(audioCtx.destination);

    noiseSource.start(now);
    noiseSource.stop(now + 0.01);
    clickOsc.start(now);
    clickOsc.stop(now + 0.015);
  } catch (e) {
    // Ignorar
  }
};

export function Laboratorio() {
  const [activeMediaType, setActiveMediaType] = useState<MediaType>('Imagen');
  const [activeModel, setActiveModel] = useState<string>(MODELS_BY_MEDIA['Imagen'][0]);
  
  const handleMediaTypeChange = (media: MediaType) => {
    setActiveMediaType(media);
    setActiveModel(MODELS_BY_MEDIA[media][0]);
    
    // Feedback sonoro "clac" mecánico
    playMechanicalClick();

    // Feedback háptico físico ultra compatible, seco y contundente de 35ms
    if ('vibrate' in navigator) {
      try {
        navigator.vibrate(35);
      } catch (err) {
        // Ignorar
      }
    }
  };
  
  const [topK, setTopK] = useState(10);
  const [isDragging, setIsDragging] = useState(false);
  const [activeSearchItem, setActiveSearchItem] = useState<number | null>(null);

  // --- Lógica de Arrastre e Interacción con el Dial ---
  const dialContainerRef = useRef<HTMLDivElement>(null);
  const [isDraggingDial, setIsDraggingDial] = useState(false);
  const [dragAngle, setDragAngle] = useState<number | null>(null);
  const lastClosestMediaRef = useRef<MediaType | null>(null);

  const getClosestMedia = (angle: number) => {
    let closest: typeof MEDIA_CONFIG[number] = MEDIA_CONFIG[0];
    let minDiff = Math.abs(angle - MEDIA_CONFIG[0].angle);
    MEDIA_CONFIG.forEach((media) => {
      const diff = Math.abs(angle - media.angle);
      if (diff < minDiff) {
        minDiff = diff;
        closest = media;
      }
    });
    return closest.id;
  };

  const currentVisualActive = (isDraggingDial && dragAngle !== null)
    ? getClosestMedia(dragAngle)
    : activeMediaType;

  const calculateAngle = (clientX: number, clientY: number) => {
    if (!dialContainerRef.current) return 0;
    const rect = dialContainerRef.current.getBoundingClientRect();
    const pivotX = rect.left + rect.width / 2;
    const pivotY = rect.bottom - 28; // El pivote está 28px arriba de la base
    
    const deltaX = clientX - pivotX;
    const deltaY = clientY - pivotY;
    
    const angleRad = Math.atan2(deltaY, deltaX);
    let angleDeg = angleRad * (180 / Math.PI);
    
    let targetAngle = angleDeg + 90; // Alinear 0 grados arriba verticalmente
    
    if (targetAngle < -180) targetAngle += 360;
    if (targetAngle > 180) targetAngle -= 360;
    
    // Rango límite para la experiencia del usuario
    return Math.max(-55, Math.min(55, targetAngle));
  };

  const handleStartDrag = (e: React.MouseEvent | React.TouchEvent) => {
    // Si hace click en una etiqueta, dejamos que el onClick nativo responda
    if ((e.target as HTMLElement).tagName === 'SPAN') return;
    
    setIsDraggingDial(true);
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    const angle = calculateAngle(clientX, clientY);
    setDragAngle(angle);
  };

  // Bloquear scroll de la página y comportamiento táctil nativo del body durante la interacción con el dial
  useEffect(() => {
    if (isDraggingDial) {
      document.body.style.overflow = 'hidden';
      document.body.style.touchAction = 'none';
    } else {
      document.body.style.overflow = '';
      document.body.style.touchAction = '';
    }
    
    return () => {
      document.body.style.overflow = '';
      document.body.style.touchAction = '';
    };
  }, [isDraggingDial]);

  useEffect(() => {
    if (!isDraggingDial) return;
    
    const handleMove = (e: MouseEvent | TouchEvent) => {
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
      const angle = calculateAngle(clientX, clientY);
      setDragAngle(angle);

      // Feedback físico (audio de engranaje + vibración marcada) al saltar entre opciones en vivo
      const closest = getClosestMedia(angle);
      if (closest !== lastClosestMediaRef.current) {
        lastClosestMediaRef.current = closest;
        
        // Micro-tick sonoro
        playGearTick();
        
        // Vibración háptica nítida de 12ms (suficiente para sentirse sin saturar el motor)
        if ('vibrate' in navigator) {
          try {
            navigator.vibrate(12);
          } catch (err) {
            // Ignorar
          }
        }
      }
    };
    
    const handleEnd = () => {
      setIsDraggingDial(false);
      if (dragAngle !== null) {
        const closestId = getClosestMedia(dragAngle);
        handleMediaTypeChange(closestId);
      }
      setDragAngle(null);
      lastClosestMediaRef.current = null;
    };
    
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleEnd);
    window.addEventListener('touchmove', handleMove, { passive: true });
    window.addEventListener('touchend', handleEnd);
    
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleEnd);
      window.removeEventListener('touchmove', handleMove);
      window.removeEventListener('touchend', handleEnd);
    };
  }, [isDraggingDial, dragAngle]);

  const activeConfig = MEDIA_CONFIG.find(m => m.id === activeMediaType) || MEDIA_CONFIG[0];


  return (
    <div className="pt-32 pb-24 px-6 max-w-7xl mx-auto min-h-screen flex flex-col">
      {/* Pivot Dial Selector */}
      <div 
        ref={dialContainerRef}
        className="relative w-full max-w-[500px] h-[250px] mx-auto mb-16 mt-4 flex justify-center overflow-visible select-none"
      >
        {/* Pivot Base */}
        <div 
          onMouseDown={handleStartDrag}
          onTouchStart={handleStartDrag}
          className="absolute bottom-0 w-14 h-14 rounded-full bg-[#1A1124]/95 border border-[#3A234A] shadow-[0_10px_30px_rgba(0,0,0,0.85),_0_0_15px_rgba(255,85,0,0.2)] flex items-center justify-center z-30 transition-all duration-300 cursor-grab active:cursor-grabbing touch-none"
        >
          {/* Glowing Orange Axis Light */}
          <div 
            className="w-5 h-5 rounded-full bg-gradient-to-br from-[#FF5500] to-[#FF8C00] border border-white/15 transition-all duration-300 relative flex items-center justify-center"
            style={{
              transform: isDraggingDial ? 'scale(1.2)' : 'scale(1)',
              boxShadow: isDraggingDial 
                ? '0 0 25px rgba(255, 85, 0, 1), inset 0 1px 3px rgba(255, 255, 255, 0.7)' 
                : '0 0 15px rgba(255, 85, 0, 0.95), inset 0 1px 2px rgba(255, 255, 255, 0.5)'
            }}
          >
            {/* Inner dynamic pulsing ring */}
            <div className={`absolute inset-0 rounded-full bg-[#FF5500]/40 transition-opacity duration-300 ${isDraggingDial ? 'animate-ping opacity-100' : 'animate-pulse opacity-75'}`} />
          </div>
        </div>

        {/* Texts on Arc (Background) */}
        {MEDIA_CONFIG.map((media) => {
          const isActive = currentVisualActive === media.id;
          return (
            <div 
              key={media.id}
              onClick={() => handleMediaTypeChange(media.id)}
              className="absolute bottom-[28px] left-1/2 cursor-pointer z-10 transition-all duration-300 flex items-center justify-center"
              style={{ 
                width: media.width,
                height: 48,
                marginLeft: -(media.width / 2),
                marginBottom: -24,
                transformOrigin: "center center",
                transform: `rotate(${media.angle}deg) translateY(-200px)` 
              }}
            >
              <span className={`block font-black text-[13px] md:text-[15px] tracking-[0.2em] uppercase transition-all duration-500 text-center ${
                isActive 
                  ? 'text-primary-container drop-shadow-[0_0_12px_rgba(255,85,0,0.8)] scale-110' 
                  : 'text-text-secondary/30 hover:text-text-secondary/60'
              }`}>
                {media.id}
              </span>
            </div>
          );
        })}

        {/* Rotating Arm (Foreground) */}
        <motion.div
          initial={false}
          animate={{ rotate: dragAngle !== null ? dragAngle : activeConfig.angle }}
          transition={isDraggingDial ? { type: "tween", duration: 0 } : { type: "spring", stiffness: 220, damping: 20 }}
          onMouseDown={handleStartDrag}
          onTouchStart={handleStartDrag}
          className="absolute bottom-[28px] origin-bottom flex flex-col items-center justify-start z-20 pointer-events-auto cursor-grab active:cursor-grabbing touch-none"
          style={{ height: 200, width: 60 }} // height is the radius
        >
          {/* Pill Head (The Hole) */}
          <motion.div
            initial={false}
            animate={{ width: activeConfig.width }}
            transition={isDraggingDial ? { type: "tween", duration: 0 } : { type: "spring", stiffness: 220, damping: 20 }}
            className="h-[48px] rounded-full border-[8px] border-[#1A1124]/95 flex items-center justify-center relative z-20"
            style={{ 
              marginTop: -24, // perfectly centers on the 200px radius
              boxShadow: '0 12px 30px rgba(0,0,0,0.7), inset 0 4px 12px rgba(0,0,0,0.9)',
              background: 'transparent' // Real hole
            }}
          >
             <div className="absolute inset-0 rounded-full border border-white/10 pointer-events-none" style={{ margin: '-8px' }}></div>
          </motion.div>
          {/* Curved Neck */}
          <motion.div 
            initial={false}
            animate={{ width: activeConfig.width - 28 }}
            transition={isDraggingDial ? { type: "tween", duration: 0 } : { type: "spring", stiffness: 220, damping: 20 }}
            className="flex-1 -mt-[2px] w-full bg-[#1A1124]/95 backdrop-blur-sm relative z-10"
            style={{
              WebkitMaskImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100' preserveAspectRatio='none'%3E%3Cpath d='M 30 100 C 30 50, 0 30, 0 0 L 100 0 C 100 30, 70 50, 70 100 Z'/%3E%3C/svg%3E")`,
              maskImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100' preserveAspectRatio='none'%3E%3Cpath d='M 30 100 C 30 50, 0 30, 0 0 L 100 0 C 100 30, 70 50, 70 100 Z'/%3E%3C/svg%3E")`,
              WebkitMaskSize: '100% 100%',
              maskSize: '100% 100%'
            }}
          >
            <div className="absolute inset-0 bg-gradient-to-b from-white/10 to-transparent opacity-50"></div>
          </motion.div>
        </motion.div>
      </div>

      {/* Category Row (Models) */}
      <section className="mb-12">
        <div className="flex md:flex-wrap gap-4 items-center justify-start md:justify-center overflow-x-auto pb-4 no-scrollbar">
          {MODELS_BY_MEDIA[activeMediaType].map((model) => (
            <button
              key={model}
              onClick={() => setActiveModel(model)}
              className={`shrink-0 px-6 py-3 rounded-full font-semibold whitespace-nowrap active:scale-95 transition-all ripple border ${
                activeModel === model
                  ? 'bg-primary-container/10 text-primary-container border-primary-container'
                  : 'btn-glass-inactive text-text-primary hover:bg-[#2D1A39]/50'
              }`}
            >
              {model}
            </button>
          ))}
        </div>
      </section>

      {activeMediaType === 'Imagen' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start flex-1 w-full">
          {/* Sidebar Controls */}
          <div className="lg:col-span-4 flex flex-col gap-8 w-full">
            {/* Configuration Card */}
            <article className="card-glass-purple rounded-[2rem] p-8 w-full">
              <h2 className="text-2xl font-bold text-primary-container mb-6">Configuración</h2>
              <div className="flex justify-between items-center mb-4">
                <label className="font-semibold text-sm text-text-secondary">Top-K Similitudes</label>
                <span className="text-primary-container font-bold text-3xl">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="50"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full mt-2"
              />
              <p className="mt-4 text-sm text-text-secondary leading-relaxed">
                Ajusta el número de imágenes similares que el motor de Red Siamesa retornará.
              </p>
            </article>

            {/* Status Card */}
            <article className="card-glass-purple rounded-[2rem] p-8 w-full">
              <div className="flex items-center gap-4 mb-4">
                <span className="w-3 h-3 bg-primary-container rounded-full animate-pulse"></span>
                <span className="font-semibold text-text-primary">Servidor GIBD: Online</span>
              </div>
              <p className="text-text-secondary text-sm">GPU A100 | Latencia 12ms | VRAM 82%</p>
            </article>
          </div>

          {/* Main Canvas Area */}
          <div className="lg:col-span-8 flex flex-col gap-8 w-full">
            {/* Upload Area */}
            <article 
              className="relative group cursor-pointer w-full"
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => { e.preventDefault(); setIsDragging(false); }}
            >
              <div className={`card-glass-purple border-2 border-dashed rounded-[3rem] p-12 min-h-[400px] flex flex-col items-center justify-center text-center transition-all duration-300 w-full ${isDragging ? 'border-primary-container bg-[#261633]/60' : 'border-border-organic/40 group-hover:border-primary-container'}`}>
                <div className="w-24 h-24 bg-primary-container/10 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <Microscope className="text-primary-container w-12 h-12" />
                </div>
                <h3 className="text-3xl font-bold text-text-primary mb-3">Sube tu Imagen de Marca</h3>
                <p className="text-text-secondary max-w-md mx-auto leading-relaxed">
                  Arrastra y suelta el archivo aquí o haz clic para explorar. Los algoritmos de visión computacional detectarán patrones únicos.
                </p>
              </div>
              <input type="file" className="absolute inset-0 opacity-0 cursor-pointer w-full h-full" />
            </article>

            {/* Primary Action Button */}
            <button className="w-full relative group bg-gradient-to-r from-primary-container to-[#ff8c00] py-6 rounded-full flex items-center justify-center gap-3 md:gap-4 hover:scale-[1.02] active:scale-95 transition-all duration-300 ripple overflow-hidden shadow-[0_0_20px_rgba(255,85,0,0.3)] hover:shadow-[0_0_30px_rgba(255,85,0,0.5)] border border-white/10">
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-in-out"></div>
              <span className="relative z-10 text-xl md:text-2xl font-black text-white uppercase tracking-tight drop-shadow-md">Buscar Similitudes (Red Siamesa)</span>
              <Search className="relative z-10 text-white w-6 h-6 md:w-8 md:h-8 drop-shadow-md" />
            </button>

            {/* Reference Results Grid */}
            <section className="mt-8 w-full">
              <h4 className="font-semibold text-sm text-text-secondary uppercase tracking-widest mb-6">Inspiración de Referencia</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
                {REFERENCE_IMAGES.map((src, i) => (
                  <div key={i} className="aspect-square card-glass-purple rounded-[1.5rem] overflow-hidden group">
                    <img src={src} className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500" alt={`Reference ${i+1}`} loading="lazy" />
                  </div>
                ))}
              </div>
            </section>
          </div>
        </div>
      )}

      {activeMediaType === 'Video' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start flex-1 w-full">
          {/* Sidebar Controls */}
          <div className="lg:col-span-4 flex flex-col gap-8 w-full">
            {/* Configuration Card */}
            <article className="card-glass-purple rounded-[2rem] p-8 w-full">
              <h2 className="text-2xl font-bold text-primary-container mb-6">Filtros Activos</h2>
              <div className="flex flex-col gap-4">
                <div className="flex justify-between items-center bg-background-base p-4 rounded-[1rem] border border-border-organic">
                  <span className="font-semibold text-sm text-text-primary">Detección de Movimiento</span>
                  <div className="w-10 h-6 bg-primary-container rounded-full relative">
                    <div className="w-4 h-4 bg-white rounded-full absolute right-1 top-1"></div>
                  </div>
                </div>
                <div className="flex justify-between items-center bg-background-base p-4 rounded-[1rem] border border-border-organic">
                  <span className="font-semibold text-sm text-text-secondary">Seguimiento Multiobjeto</span>
                  <div className="w-10 h-6 bg-[#0d0712]/50 border border-border-organic/40 rounded-full relative">
                    <div className="w-4 h-4 bg-text-secondary rounded-full absolute left-1 top-1"></div>
                  </div>
                </div>
              </div>
            </article>

            {/* Status Card */}
            <article className="card-glass-purple rounded-[2rem] p-8 w-full">
              <div className="flex items-center gap-4 mb-4">
                <span className="w-3 h-3 bg-primary-container rounded-full animate-pulse"></span>
                <span className="font-semibold text-text-primary">Servidor GIBD: Online</span>
              </div>
              <p className="text-text-secondary text-sm">GPU A100 | Latencia 18ms | VRAM 94%</p>
            </article>
          </div>

          {/* Main Canvas Area */}
          <div className="lg:col-span-8 flex flex-col gap-8 w-full">
            {/* Upload Area for Video/Audio */}
            <article 
              className="relative group cursor-pointer w-full"
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => { e.preventDefault(); setIsDragging(false); }}
            >
              <div className={`card-glass-purple border-2 border-dashed rounded-[3rem] p-12 min-h-[400px] flex flex-col items-center justify-center text-center transition-all duration-300 w-full ${isDragging ? 'border-primary-container bg-[#261633]/60' : 'border-border-organic/40 group-hover:border-primary-container'}`}>
                <div className="w-24 h-24 bg-primary-container/10 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <Video className="text-primary-container w-12 h-12" />
                </div>
                <h3 className="text-3xl font-bold text-text-primary mb-3">Sube tu Archivo Multimedia</h3>
                <p className="text-text-secondary max-w-md mx-auto leading-relaxed">
                  Soporte para MP4, AVI, o extractos de audio. Los modelos buscarán secuencias temporales anómalas o patrones específicos.
                </p>
              </div>
              <input type="file" className="absolute inset-0 opacity-0 cursor-pointer w-full h-full" accept="video/*" />
            </article>

            <button className="w-full relative group bg-gradient-to-r from-primary-container to-[#ff8c00] py-6 rounded-full flex items-center justify-center gap-3 md:gap-4 hover:scale-[1.02] active:scale-95 transition-all duration-300 ripple overflow-hidden shadow-[0_0_20px_rgba(255,85,0,0.3)] hover:shadow-[0_0_30px_rgba(255,85,0,0.5)] border border-white/10">
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-in-out"></div>
              <span className="relative z-10 text-xl md:text-2xl font-black text-white uppercase tracking-tight drop-shadow-md">Procesar Secuencia Temporal</span>
              <Sparkles className="relative z-10 text-white w-6 h-6 md:w-8 md:h-8 drop-shadow-md" />
            </button>
          </div>
        </div>
      )}

      {activeMediaType === 'Texto' && (
        <div className="flex flex-col md:flex-row gap-6 h-[700px] w-full">
          {/* Left Sidebar */}
          <aside className="w-full md:w-80 flex flex-col gap-4 overflow-hidden hidden md:flex shrink-0">
            <div className="px-4">
              <h2 className="text-xl font-bold text-primary-container mb-2">Colección Documental</h2>
              <div className="h-1 w-12 bg-primary-container rounded-full"></div>
            </div>
            
            <div className="flex-1 overflow-y-auto custom-scrollbar space-y-3 pr-2">
              {SIDEBAR_ITEMS.map((item) => (
                <div 
                  key={item.id}
                  onClick={() => setActiveSearchItem(item.id)}
                  className={`p-6 rounded-full border cursor-pointer transition-all group flex items-center gap-4 ${item.opacity || ''} ${
                    activeSearchItem === item.id 
                      ? 'bg-primary-container/10 border-primary-container' 
                      : 'btn-glass-inactive border-transparent hover:border-primary-container'
                  }`}
                >
                  <item.icon className="text-primary-container w-6 h-6 shrink-0" />
                  <span className="font-semibold text-sm truncate">{item.label}</span>
                </div>
              ))}
            </div>
          </aside>

          {/* Main Content Area */}
          <section className="flex-1 flex flex-col card-glass-purple rounded-[3rem] overflow-hidden relative min-h-[500px] md:min-h-0 w-full">
            {/* Empty State */}
            <div className="flex-1 flex flex-col items-center justify-center p-8 md:p-12 text-center h-full pb-32">
              <div className="w-32 h-32 bg-primary-container/10 rounded-full flex items-center justify-center mb-8 animate-pulse shrink-0">
                <Sparkles className="w-16 h-16 text-primary-container" />
              </div>
              <h3 className="text-2xl md:text-[28px] font-bold max-w-lg leading-tight text-text-primary">
                Escribe una consulta en lenguaje natural
              </h3>
              <p className="mt-4 text-text-secondary max-w-md leading-relaxed">
                La red neuronal semántica del GIBD analizará millones de registros en la base de datos distribuida para encontrar correlaciones ocultas en papers o documentos jurídicos.
              </p>
            </div>

            {/* Bottom Input Section */}
            <div className="p-4 md:p-8 bg-black/40 backdrop-blur-md absolute bottom-0 w-full border-t border-border-organic">
              <div className="flex flex-col md:flex-row gap-4 items-center">
                <div className="flex-1 w-full bg-background-base rounded-full border border-border-organic px-8 py-4 flex items-center gap-4 focus-within:border-primary-container transition-colors">
                  <TerminalSquare className="text-text-secondary w-6 h-6 shrink-0" />
                  <input 
                    type="text" 
                    placeholder="Busca por patrones o conceptos (NLP)..." 
                    className="bg-transparent border-none outline-none focus:ring-0 w-full text-text-primary placeholder:text-text-secondary font-semibold"
                  />
                </div>
                <button className="w-full md:w-auto relative group bg-gradient-to-r from-primary-container to-[#ff8c00] text-white font-bold text-lg px-8 md:px-12 py-4 rounded-full whitespace-nowrap hover:scale-105 active:scale-95 transition-all duration-300 ripple overflow-hidden shadow-[0_0_15px_rgba(255,85,0,0.3)] hover:shadow-[0_0_25px_rgba(255,85,0,0.5)] border border-white/10">
                  <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-in-out"></div>
                  <span className="relative z-10 drop-shadow-md">Buscar (NLP)</span>
                </button>
              </div>
            </div>
          </section>
        </div>
      )}

      {activeMediaType === 'Sonido' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start flex-1 w-full animate-fade-in">
          {/* Sidebar Controls */}
          <div className="lg:col-span-4 flex flex-col gap-8 w-full">
            {/* Configuration Card */}
            <article className="card-glass-purple rounded-[2rem] p-8 w-full">
              <h2 className="text-2xl font-bold text-primary-container mb-6">Configuración de Audio</h2>
              <div className="flex justify-between items-center mb-4">
                <label className="font-semibold text-sm text-text-secondary">Umbral de Ruido (dB)</label>
                <span className="text-primary-container font-bold text-3xl">-35</span>
              </div>
              <input
                type="range"
                min="-60"
                max="0"
                defaultValue="-35"
                className="w-full mt-2"
              />
              <p className="mt-4 text-sm text-text-secondary leading-relaxed">
                Filtra frecuencias de fondo para potenciar la precisión de la inferencia del modelo.
              </p>
            </article>

            {/* Status Card */}
            <article className="card-glass-purple rounded-[2rem] p-8 w-full">
              <div className="flex items-center gap-4 mb-4">
                <span className="w-3 h-3 bg-primary-container rounded-full animate-pulse"></span>
                <span className="font-semibold text-text-primary">Servidor GIBD: Online</span>
              </div>
              <p className="text-text-secondary text-sm">GPU A100 | Latencia 9ms | VRAM 64%</p>
            </article>
          </div>

          {/* Main Canvas Area */}
          <div className="lg:col-span-8 flex flex-col gap-8 w-full">
            {/* Upload Area for Audio */}
            <article 
              className="relative group cursor-pointer w-full"
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => { e.preventDefault(); setIsDragging(false); }}
            >
              <div className={`card-glass-purple border-2 border-dashed rounded-[3rem] p-12 min-h-[400px] flex flex-col items-center justify-center text-center transition-all duration-300 w-full ${isDragging ? 'border-primary-container bg-[#261633]/60' : 'border-border-organic/40 group-hover:border-primary-container'}`}>
                <div className="w-24 h-24 bg-primary-container/10 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <AudioLines className="text-primary-container w-12 h-12" />
                </div>
                <h3 className="text-3xl font-bold text-text-primary mb-3">Sube tu Archivo de Sonido</h3>
                <p className="text-text-secondary max-w-md mx-auto leading-relaxed">
                  Formatos soportados: MP3, WAV. Se generará un espectrograma y se ejecutará una red neuronal recurrente.
                </p>
                {/* Liquid Waveform */}
                <div className="flex items-center gap-1.5 mt-8 h-12 justify-center">
                  {[...Array(16)].map((_, i) => (
                    <motion.div
                      key={i}
                      className="w-1.5 bg-primary-container rounded-full shadow-[0_0_8px_rgba(255,85,0,0.5)]"
                      animate={{
                        height: [12, 48, 16, 32, 12][i % 5],
                      }}
                      transition={{
                        duration: 1.2 + (i % 3) * 0.2,
                        repeat: Infinity,
                        repeatType: "reverse",
                        ease: "easeInOut"
                      }}
                    />
                  ))}
                </div>
              </div>
              <input type="file" className="absolute inset-0 opacity-0 cursor-pointer w-full h-full" accept="audio/*" />
            </article>

            <button className="w-full relative group bg-gradient-to-r from-primary-container to-[#ff8c00] py-6 rounded-full flex items-center justify-center gap-3 md:gap-4 hover:scale-[1.02] active:scale-95 transition-all duration-300 ripple overflow-hidden shadow-[0_0_20px_rgba(255,85,0,0.3)] hover:shadow-[0_0_30px_rgba(255,85,0,0.5)] border border-white/10">
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-in-out"></div>
              <span className="relative z-10 text-xl md:text-2xl font-black text-white uppercase tracking-tight drop-shadow-md">Analizar Espectrograma</span>
              <Sparkles className="relative z-10 text-white w-6 h-6 md:w-8 md:h-8 drop-shadow-md" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
