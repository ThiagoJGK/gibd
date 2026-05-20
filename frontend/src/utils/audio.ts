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

// Genera un sonido gomoso (rubbery snap) para el selector de cinta.
// Simula el golpe suave de una banda elástica: ruido blanco filtrado con una
// frecuencia de resonancia alta y decaimiento muy rápido.
export const playRubberSnap = () => {
  try {
    const audioCtx = getAudioCtx();
    if (!audioCtx) return;
    const now = audioCtx.currentTime;

    const duration = 0.22; // 220ms de slide orgánico

    const osc = audioCtx.createOscillator();
    const subOsc = audioCtx.createOscillator();
    const filter = audioCtx.createBiquadFilter();
    const gainNode = audioCtx.createGain();

    // 1. Oscilador principal (Diente de sierra para armónicos cálidos de cuerda)
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(150, now); // Comienza en Sol2/La2
    osc.frequency.exponentialRampToValueAtTime(75, now + duration); // Hace un slide de octava hacia abajo

    // 2. Sub-oscilador (Triángulo para darle cuerpo y peso de bajo real)
    subOsc.type = 'triangle';
    subOsc.frequency.setValueAtTime(150, now);
    subOsc.frequency.exponentialRampToValueAtTime(75, now + duration);

    // 3. Filtro Paso-Bajo Dinámico (Simula el apagado de la cuerda y quita lo digital)
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(500, now); // Comienza un poco brillante
    filter.frequency.exponentialRampToValueAtTime(90, now + duration); // Se apaga imitando la fricción de madera/goma
    filter.Q.setValueAtTime(3, now); // Resonancia analógica sutil

    // 4. Envolvente de Volumen (Bajo volumen de pulsación)
    gainNode.gain.setValueAtTime(0.001, now);
    gainNode.gain.linearRampToValueAtTime(0.08, now + 0.008); // Volumen bajado a 0.08 (de 0.24)
    gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Conexiones
    osc.connect(filter);
    subOsc.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    osc.start(now);
    subOsc.start(now);
    osc.stop(now + duration);
    subOsc.stop(now + duration);
  } catch (e) {}
};

// Genera un sonido de fricción orgánico y ultra-corto (un roce físico sutil)
// al arrastrar la cinta por los engranajes del selector.
let lastTickTime = 0;

export const playFrictionTick = () => {
  try {
    const audioCtx = getAudioCtx();
    if (!audioCtx) return;
    const now = audioCtx.currentTime;

    // Evita superposiciones molestas y suma de volumen (mínimo 60ms entre cada sonido)
    if (now - lastTickTime < 0.060) return;
    lastTickTime = now;

    const osc = audioCtx.createOscillator();
    const filter = audioCtx.createBiquadFilter();
    const gainNode = audioCtx.createGain();

    // Frecuencias medias (550Hz -> 220Hz) audibles en cualquier celular
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(550, now);
    osc.frequency.exponentialRampToValueAtTime(220, now + 0.015); // Barrido rápido

    // Filtro sutil para conservar calidez acústica (tipo madera/plástico duro)
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(600, now);
    filter.Q.setValueAtTime(1.5, now);

    // Envolvente súper corta (15ms) para que no sature ni ensucie al deslizar rápido
    gainNode.gain.setValueAtTime(0.001, now);
    gainNode.gain.linearRampToValueAtTime(0.08, now + 0.003); // Transitorio instantáneo
    gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.015);

    osc.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    osc.start(now);
    osc.stop(now + 0.015);
  } catch (e) {}
};





