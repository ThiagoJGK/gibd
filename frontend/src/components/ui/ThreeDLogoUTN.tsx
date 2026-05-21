import React, { useEffect, useRef } from 'react';

interface Point3D {
  x0: number; // base x coordinate
  y0: number; // base y coordinate
  z0: number; // base z coordinate
  dx: number; // dispersion vector x
  dy: number; // dispersion vector y
  dz: number; // dispersion vector z
  baseSize: number;
}

export const ThreeDLogoUTN: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const scrollPercentRef = useRef<number>(0);
  const mouseRef = useRef<{ x: number | null; y: number | null }>({ x: null, y: null });
  const isVisibleRef = useRef<boolean>(true);
  const animationFrameIdRef = useRef<number | null>(null);
  
  // Point cloud generation (precomputed once)
  const pointsRef = useRef<Point3D[]>([]);

  // 1. Generate points on mount
  useEffect(() => {
    const points: Point3D[] = [];

    // Auxiliary to add a point with random z thickness and dispersion vector
    const addPoint = (x: number, y: number, zRange: number = 10) => {
      const z = (Math.random() - 0.5) * zRange;
      
      // Calculate explosive direction radial from logo center
      const len = Math.hypot(x, y, z) || 1;
      const dx = (x / len) * (1.2 + Math.random() * 1.8);
      const dy = (y / len) * (1.2 + Math.random() * 1.8);
      const dz = (z / len) * (1.2 + Math.random() * 1.8);
      
      const baseSize = 1.0 + Math.random() * 1.0; // Random dot size between 1px and 2px

      points.push({ x0: x, y0: y, z0: z, dx, dy, dz, baseSize });
    };

    // Modeling "La Araña" UTN Logo mathematically with correct SVG proportions:
    // Scale: Height of central column is 60, outer arch radius is 24, inner radius is 16

    // A. Central Vertical Column:
    // x in [-4.75, 4.75], y in [-29.75, 29.75]
    const vertPointsCount = 450;
    for (let i = 0; i < vertPointsCount; i++) {
      const x = (Math.random() - 0.5) * 9.5;
      const y = (Math.random() - 0.5) * 59.5;
      addPoint(x, y);
    }

    // B. Central Horizontal Bar:
    // x in [-24.65, 24.65], y in [-4.65, 4.65]
    const horizPointsCount = 350;
    for (let i = 0; i < horizPointsCount; i++) {
      const x = (Math.random() - 0.5) * 49.3;
      const y = (Math.random() - 0.5) * 9.3;
      addPoint(x, y);
    }

    // C. Top Arch Shape (Open ends pointing UPWARDS, peak pointing DOWNWARDS towards center):
    // Radii: [16, 24], Center shifted to y = -29.2
    // y = -29.2 + r * Math.sin(angle), angle in [0, pi]
    const topArchPointsCount = 500;
    for (let i = 0; i < topArchPointsCount; i++) {
      const r = 16 + Math.random() * 8; // [16, 24]
      const angle = Math.random() * Math.PI; // [0, pi]
      const x = r * Math.cos(angle);
      const y = -29.2 + r * Math.sin(angle); // curved peak points down towards center
      addPoint(x, y);
    }

    // D. Bottom Arch Shape (Open ends pointing DOWNWARDS, peak pointing UPWARDS towards center):
    // Radii: [16, 24], Center shifted to y = 29.2
    // y = 29.2 - r * Math.sin(angle), angle in [0, pi]
    const bottomArchPointsCount = 500;
    for (let i = 0; i < bottomArchPointsCount; i++) {
      const r = 16 + Math.random() * 8; // [16, 24]
      const angle = Math.random() * Math.PI; // [0, pi]
      const x = r * Math.cos(angle);
      const y = 29.2 - r * Math.sin(angle); // curved peak points up towards center
      addPoint(x, y);
    }

    pointsRef.current = points;
  }, []);

  // 2. Track global scroll progress in a mutable Ref (prevents React re-renders & animation restarts)
  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const maxScroll = 500; // dilutes completely at 500px scroll
      const progress = Math.min(scrollY / maxScroll, 1.5);
      scrollPercentRef.current = progress;
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // 3. Intersection Observer to check if off-screen (0% CPU optimization)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        isVisibleRef.current = entry.isIntersecting;
      },
      { threshold: 0.01 }
    );

    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // 4. Color interpolation based on depth Z-depth (from front orange to back deep purple)
  const getDepthColor = (t: number): string => {
    // t is 0 (back) to 1 (front)
    t = Math.max(0, Math.min(1, t));
    
    let r, g, b;
    if (t > 0.5) {
      // Interpolate: Middle (#A855F7) -> Front (#FF5500)
      // rgb(168, 85, 247) -> rgb(255, 85, 0)
      const factor = (t - 0.5) * 2;
      r = Math.round(168 + (255 - 168) * factor);
      g = Math.round(85 + (85 - 85) * factor);
      b = Math.round(247 + (0 - 247) * factor);
    } else {
      // Interpolate: Back (#120424) -> Middle (#A855F7)
      // rgb(18, 4, 36) -> rgb(168, 85, 247)
      const factor = t * 2;
      r = Math.round(18 + (168 - 18) * factor);
      g = Math.round(4 + (85 - 4) * factor);
      b = Math.round(36 + (247 - 36) * factor);
    }
    return `rgb(${r}, ${g}, ${b})`;
  };

  // 5. Main animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let angleY = 0;
    let angleX = 0.2; // slight downward tilt for better isometric depth
    let lastTime = performance.now();

    const render = () => {
      const scrollPercent = scrollPercentRef.current;
      // If component is off-screen or completely faded, pause loop
      if (!isVisibleRef.current || scrollPercent >= 1.4) {
        animationFrameIdRef.current = requestAnimationFrame(render);
        return;
      }

      const now = performance.now();
      const deltaTime = (now - lastTime) / 1000;
      lastTime = now;

      // Slow Y-axis rotation (0.15 rads per second)
      angleY += 0.20 * deltaTime;
      // Very subtle rocking on X-axis for organic look
      angleX = 0.25 + Math.sin(now * 0.001) * 0.05;

      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Setup glowing holographic points style using additive blending
      ctx.globalCompositeOperation = 'screen';

      const cx = width / 2;
      const cy = height / 2;
      const cameraDistance = 140;
      const focalLength = 1200; // perspective zoom factor (increased to 1200 to fully occupy the right side)
      
      const points = pointsRef.current;
      const mouse = mouseRef.current;

      // Dispersion scale (radial explosion) based on scroll progress
      // The explosion is softened to make the scroll dispersion feel elegant and smooth
      const dispersionMultiplier = Math.pow(scrollPercent, 1.2) * 90;
      const globalOpacity = Math.max(0, 1 - Math.pow(scrollPercent, 1.2));

      // Early exit if points are completely faded out (100% CPU saving)
      if (globalOpacity <= 0) {
        animationFrameIdRef.current = requestAnimationFrame(render);
        return;
      }

      // Calculate trig values once for the loop
      const cosY = Math.cos(angleY);
      const sinY = Math.sin(angleY);
      const cosX = Math.cos(angleX);
      const sinX = Math.sin(angleX);


      for (let i = 0; i < points.length; i++) {
        const p = points[i];

        // A. Scroll-based radial explosion displacement
        let x = p.x0 + p.dx * dispersionMultiplier;
        let y = p.y0 + p.dy * dispersionMultiplier;
        let z = p.z0 + p.dz * dispersionMultiplier;

        // B. Apply 3D Rotation (Y-axis first, then X-axis)
        // Y-axis rotation
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;
        
        // X-axis rotation
        const y2 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;

        // C. Perspective Projection
        const dist = cameraDistance - z2;
        if (dist <= 1) continue; // behind camera check

        let screenX = cx + (x1 * focalLength) / dist;
        let screenY = cy + (y2 * focalLength) / dist;

        // D. Calculate normalized Z depth for color gradient
        // Rotated Z2 lies within [-25, 25] bounds
        const normalizedDepth = (z2 + 25) / 50;

        // E. Mouse Proximity Repulsion & Ignition
        let color = getDepthColor(normalizedDepth);
        let size = p.baseSize;
        let finalOpacity = globalOpacity;
        let force = 0;

        if (mouse.x !== null && mouse.y !== null) {
          const dxMouse = screenX - mouse.x;
          const dyMouse = screenY - mouse.y;
          const distMouse = Math.hypot(dxMouse, dyMouse);
          const maxDistMouse = 160; // active hover radius (scaled to match the massive logo size)

          if (distMouse < maxDistMouse) {
            // Proximity force (1 at cursor, 0 at boundary)
            force = (maxDistMouse - distMouse) / maxDistMouse;
            
            // 1. Repulsion (push away from mouse cursor - softened for a gentle, organic sway)
            if (distMouse > 0) {
              screenX += (dxMouse / distMouse) * force * 15; // scaled slightly for the larger logo
              screenY += (dyMouse / distMouse) * force * 15;
            }

            // 2. Point Ignition (subtle size scale up to +75% instead of +220% to keep dots sharp and dense)
            size *= (1 + force * 0.75);

            // 3. Ignite Color (blend from depth color to white/yellow core)
            const rgbMatch = color.match(/\d+/g);
            if (rgbMatch) {
              const dr = parseInt(rgbMatch[0], 10);
              const dg = parseInt(rgbMatch[1], 10);
              const db = parseInt(rgbMatch[2], 10);
              
              // Target is bright yellow (255, 230, 80) near edges, pure white (255, 255, 255) at the center
              const targetR = 255;
              const targetG = Math.round(220 + 35 * force);
              const targetB = Math.round(100 + 155 * force);

              const r = Math.round(dr + (targetR - dr) * force);
              const g = Math.round(dg + (targetG - dg) * force);
              const b = Math.round(db + (targetB - db) * force);
              color = `rgb(${r}, ${g}, ${b})`;
            }

            // Maximize opacity for active glowing dots
            finalOpacity = Math.max(finalOpacity, force * 0.9 * globalOpacity);
          }
        }

        // Smooth Vignette/Edge fading to prevent abrupt box clipping at the canvas borders
        const fadePadding = 90; // Pixels near the edge where points start to fade
        let borderFade = 1;
        if (screenX < fadePadding) {
          borderFade *= Math.max(0, screenX / fadePadding);
        } else if (screenX > width - fadePadding) {
          borderFade *= Math.max(0, (width - screenX) / fadePadding);
        }
        if (screenY < fadePadding) {
          borderFade *= Math.max(0, screenY / fadePadding);
        } else if (screenY > height - fadePadding) {
          borderFade *= Math.max(0, (height - screenY) / fadePadding);
        }
        finalOpacity *= borderFade;

        // F. Render Point
        // Draw soft glowing halo for active points close to the mouse
        if (force > 0) {
          ctx.beginPath();
          ctx.arc(screenX, screenY, size * 2.6, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.globalAlpha = finalOpacity * 0.24 * force; // soft bloom opacity
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(screenX, screenY, size, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.globalAlpha = finalOpacity;
        ctx.fill();
      }

      animationFrameIdRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, []); // Run animation loop once on mount (scroll updates via Ref avoid teardowns/restarts)

  // 6. Handle Mouse Interactions (Global window listener to bypass layer blockage)
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas || !isVisibleRef.current) return;

      const rect = canvas.getBoundingClientRect();
      const xLocal = e.clientX - rect.left;
      const yLocal = e.clientY - rect.top;

      // Scale standard client coords to match canvas high-res layout dimensions
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      // Allow a buffer area (100px padding) so the glow is active even just outside the canvas border
      const padding = 100;
      if (
        xLocal >= -padding &&
        xLocal <= rect.width + padding &&
        yLocal >= -padding &&
        yLocal <= rect.height + padding
      ) {
        mouseRef.current = {
          x: xLocal * scaleX,
          y: yLocal * scaleY,
        };
      } else {
        mouseRef.current = { x: null, y: null };
      }
    };

    const handleGlobalMouseLeave = () => {
      mouseRef.current = { x: null, y: null };
    };

    window.addEventListener('mousemove', handleGlobalMouseMove, { passive: true });
    document.addEventListener('mouseleave', handleGlobalMouseLeave);

    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      document.removeEventListener('mouseleave', handleGlobalMouseLeave);
    };
  }, []);

  return (
    <div 
      ref={containerRef}
      className="w-full h-full flex items-center justify-center pointer-events-none"
      style={{ minHeight: '720px' }}
    >
      <canvas
        ref={canvasRef}
        width={760}
        height={760}
        className="w-[760px] h-[760px] select-none max-w-full drop-shadow-[0_0_45px_rgba(255,85,0,0.2)] pointer-events-none"
      />
    </div>
  );
};

export default ThreeDLogoUTN;
