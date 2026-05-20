import { useState, useEffect, useRef } from 'react';
import { motion, useMotionValue, useTransform, animate, AnimatePresence } from 'motion/react';
import { playFrictionTick } from '../../utils/audio';

const ITEM_WIDTH = 200;

interface RibbonSelectorProps {
  options: string[];
  activeOption: string;
  onChange: (option: string) => void;
}

export function RibbonSelector({ options, activeOption, onChange }: RibbonSelectorProps) {
  const x = useMotionValue(0);
  const activeIndex = options.indexOf(activeOption);
  const lastIndexRef = useRef(activeIndex);
  
  const [showHint, setShowHint] = useState(() => {
    try { return !localStorage.getItem('ribbon-swiped'); }
    catch { return false; }
  });

  // Track activeIndex changes
  useEffect(() => {
    lastIndexRef.current = activeIndex;
  }, [activeIndex]);

  // Haptic friction ticks on crossing midpoints during drag
  useEffect(() => {
    const unsubscribe = x.on("change", (latest) => {
      const index = Math.round(-latest / ITEM_WIDTH);
      if (index !== lastIndexRef.current && index >= 0 && index < options.length) {
        lastIndexRef.current = index;
        playFrictionTick();
        if (typeof navigator !== 'undefined' && navigator.vibrate) {
          navigator.vibrate(18); // Strong tactile friction tick!
        }
      }
    });
    return () => unsubscribe();
  }, [x, options.length]);

  // Sync external activeOption changes to the ribbon position
  useEffect(() => {
    if (activeIndex === -1) return;
    const targetX = -(activeIndex * ITEM_WIDTH);
    animate(x, targetX, { type: 'spring', stiffness: 300, damping: 25 });
  }, [activeOption, activeIndex, x]);

  const handleItemClick = (opt: string) => {
    if (showHint) {
      setShowHint(false);
      try { localStorage.setItem('ribbon-swiped', '1'); } catch {}
    }
    if (typeof navigator !== 'undefined' && navigator.vibrate) {
      navigator.vibrate(12); // Vibración corta y táctil gomosa
    }
    playFrictionTick();
    onChange(opt);
  };

  const handleDragEnd = (_event: any, info: any) => {
    // Dismiss hint forever on first drag
    if (showHint) {
      setShowHint(false);
      try { localStorage.setItem('ribbon-swiped', '1'); } catch {}
    }
    if (typeof navigator !== 'undefined' && navigator.vibrate) {
      navigator.vibrate(15); // Vibración un poquito más marcada para la inercia
    }
    // Project where it would land based on velocity
    const projectedX = x.get() + info.velocity.x * 0.15; 
    let targetIndex = Math.round(-projectedX / ITEM_WIDTH);
    targetIndex = Math.max(0, Math.min(targetIndex, options.length - 1));
    playFrictionTick();
    onChange(options[targetIndex]);
  };

  return (
    <div 
      className="relative w-full mx-auto h-[140px] flex items-center justify-center overflow-hidden touch-pan-y"
      style={{ 
        WebkitMaskImage: 'linear-gradient(to right, transparent, black 15%, black 85%, transparent)',
        maskImage: 'linear-gradient(to right, transparent, black 15%, black 85%, transparent)'
      }}
    >
      {/* Central Fixed Pill (Hollow Lense) */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-20 flex flex-col items-center">
         {/* Solid Rounded Dark Triangle */}
         <div className="absolute left-1/2 top-0 -translate-x-1/2 -translate-y-[85%]">
           <svg width="16" height="12" viewBox="0 0 16 12" fill="none" xmlns="http://www.w3.org/2000/svg">
             <path d="M8 10L2 2H14L8 10Z" fill="#1A1124" stroke="#3A234A" strokeWidth="2" strokeLinejoin="round" />
           </svg>
         </div>
         {/* Hollow Pill with Dial Aesthetic */}
         <div 
            className="rounded-full border-[6px] border-[#1A1124]/95 relative"
            style={{ 
              width: ITEM_WIDTH, 
              height: 56,
              boxShadow: '0 12px 30px rgba(0,0,0,0.7), inset 0 4px 12px rgba(0,0,0,0.9)',
              background: 'transparent'
            }}
         >
           <div className="absolute inset-0 rounded-full border border-primary-container/60 shadow-[inset_0_0_15px_rgba(255,85,0,0.2),_0_0_10px_rgba(255,85,0,0.3)] pointer-events-none" style={{ margin: '-6px' }}></div>
         </div>
      </div>
      
      {/* Draggable Ribbon */}
      <motion.div 
        className="flex items-center absolute left-1/2 h-[56px] top-1/2 -translate-y-1/2 z-10"
        style={{ x, marginLeft: -(ITEM_WIDTH / 2) }}
        drag="x"
        dragDirectionLock={true}
        dragConstraints={{ left: -((options.length - 1) * ITEM_WIDTH), right: 0 }}
        dragElastic={0.15}
        onDragEnd={handleDragEnd}
      >
        {options.map((opt, i) => (
           <RibbonItem 
              key={opt} 
              option={opt} 
              index={i} 
              currentX={x} 
              itemWidth={ITEM_WIDTH} 
              onClick={() => handleItemClick(opt)}
           />
        ))}
      </motion.div>
      {/* Swipe Hint */}
      <AnimatePresence>
        {showHint && (
          <motion.div
            className="absolute bottom-1 left-1/2 -translate-x-1/2 pointer-events-none z-30 flex items-center gap-1"
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 0.45, 0.45, 0] }}
            transition={{ duration: 2.5, times: [0, 0.2, 0.8, 1], repeat: Infinity, repeatDelay: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.3 } }}
          >
            <span className="text-[10px] uppercase tracking-[0.3em] text-text-secondary font-semibold select-none">← desliza →</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function RibbonItem({ option, index, currentX, itemWidth, onClick }: any) {
  // Exact x-coordinate where this item is perfectly centered in the lens
  const myCenterPosition = -(index * itemWidth);
  
  // Calculate raw distance from center
  const distance = useTransform(currentX, (val: number) => Math.abs(val - myCenterPosition));
  
  // Map distance to visual properties
  // When perfectly centered (distance 0) -> fully visible, primary color
  // When outside the pill (distance > itemWidth * 0.4) -> dimmed, smaller
  const opacity = useTransform(distance, [0, itemWidth * 0.4, itemWidth * 1.5], [1, 0.8, 0.5]);
  const scale = useTransform(distance, [0, itemWidth * 0.5], [1, 0.9]);
  const color = useTransform(distance, [0, itemWidth * 0.2, itemWidth * 0.5], ['#FF5500', '#FF5500', '#A19DAB']); 
  const textShadow = useTransform(
    distance, 
    [0, itemWidth * 0.2], 
    ['0px 0px 15px rgba(255,85,0,0.8)', '0px 0px 0px rgba(0,0,0,0)']
  );

  return (
    <motion.div 
      onClick={onClick}
      className="flex items-center justify-center font-black uppercase tracking-widest text-[12px] md:text-[14px] cursor-pointer select-none text-center px-4"
      style={{ width: itemWidth, opacity, scale, color, textShadow }}
    >
      {option}
    </motion.div>
  );
}
