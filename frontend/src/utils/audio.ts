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

// Genera un feedback sonoro líquido de gota de agua física de alta fidelidad
export const playWaterDrip = () => {
  try {
    const audioCtx = getAudioCtx();
    if (!audioCtx) return;

    const now = audioCtx.currentTime;

    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    osc.type = 'sine';

    // Curva exponencial ascendente (burbuja/gota de agua ascendente)
    osc.frequency.setValueAtTime(550, now);
    osc.frequency.exponentialRampToValueAtTime(1350, now + 0.12);

    // Envolvente con ataque suave instantáneo (12ms) y decaimiento rápido (120ms)
    gainNode.gain.setValueAtTime(0.0001, now);
    gainNode.gain.linearRampToValueAtTime(0.12, now + 0.012);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);

    osc.start(now);
    osc.stop(now + 0.12);
  } catch (e) {
    // Ignorar si el navegador bloquea audio antes de la interacción inicial
  }
};
