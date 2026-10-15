// Cache global de AudioContext para reducir la presión del recolector de basura (GC) y consumo de RAM
let cachedAudioCtx: AudioContext | null = null;

export const getAudioCtx = (): AudioContext | null => {
  if (typeof window === 'undefined') return null;
  if (!cachedAudioCtx) {
    const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
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

// Audio feedback físico de alta fidelidad: Clac mecánico (Cherry MX / Rotary Switch)
export const playMechanicalClick = () => {
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
export const playGearTick = () => {
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
    // Ignorar si el navegador bloquea audio antes de la interacción
  }
};
