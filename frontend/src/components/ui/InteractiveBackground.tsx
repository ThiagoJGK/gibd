import { useEffect, useRef } from 'react';

export function InteractiveBackground() {
  const blobRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let currentX = window.innerWidth / 2;
    let currentY = window.innerHeight / 2;
    let targetX = window.innerWidth / 2;
    let targetY = window.innerHeight / 2;
    let isMobile = window.innerWidth <= 768;

    const handleResize = () => {
      isMobile = window.innerWidth <= 768;
    };
    window.addEventListener('resize', handleResize);

    const handleMouseMove = (e: MouseEvent) => {
      if (isMobile) return;
      targetX = e.clientX;
      targetY = e.clientY;
    };

    window.addEventListener('mousemove', handleMouseMove);

    let animationFrameId: number;
    let randomTimer: ReturnType<typeof setTimeout>;

    const pickRandomTarget = () => {
      if (isMobile) {
        targetX = Math.random() * window.innerWidth;
        targetY = Math.random() * window.innerHeight;
      }
      randomTimer = setTimeout(pickRandomTarget, 2000 + Math.random() * 3000);
    };

    pickRandomTarget();

    const loop = () => {
      const speed = isMobile ? 0.005 : 0.05;
      currentX += (targetX - currentX) * speed;
      currentY += (targetY - currentY) * speed;

      if (blobRef.current) {
        blobRef.current.style.transform = `translate(${currentX}px, ${currentY}px)`;
      }

      animationFrameId = requestAnimationFrame(loop);
    };

    loop();

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      clearTimeout(randomTimer);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
      {/* Estela (Blob) que persigue al cursor */}
      <div 
        ref={blobRef}
        className="absolute left-[-250px] top-[-250px] w-[500px] h-[500px] rounded-full bg-primary-container opacity-20 blur-[120px] will-change-transform mix-blend-screen"
      />
      
      {/* Grilla superior */}
      <div 
        className="absolute inset-0" 
        style={{
          backgroundImage: 'linear-gradient(to right, rgba(255, 255, 255, 0.08) 1px, transparent 1px), linear-gradient(to bottom, rgba(255, 255, 255, 0.08) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
          maskImage: 'linear-gradient(to bottom, rgba(0,0,0,1) 60%, rgba(0,0,0,0) 100%)',
          WebkitMaskImage: 'linear-gradient(to bottom, rgba(0,0,0,1) 60%, rgba(0,0,0,0) 100%)'
        }}
      />
    </div>
  );
}
