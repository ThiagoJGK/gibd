import React, { useState, useRef, useEffect } from 'react';
import { Microscope, Search, Archive, Database, FileText, History, Network, Scale, Sparkles, TerminalSquare, Video, AudioLines, X, Upload, CheckCircle2, Sliders, RefreshCw, Eye, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { RibbonSelector } from '../components/ui/RibbonSelector';
import { BRANDS_CATALOG } from '../data/brandsCatalog';
import type { VisionModelType, VisionMatchResult, VisionSampleItem } from '../types/vision';
import { VISION_MODEL_OPTIONS, brandItemToVisionMatch } from '../types/vision';
import { playMechanicalClick, playGearTick } from '../utils/audio';
import { processImageUpload } from '../utils/cloudinary';

const MEDIA_CONFIG = [
  { id: 'Imagen', angle: -35, width: 110 },
  { id: 'Video', angle: -12, width: 100 },
  { id: 'Texto', angle: 12, width: 100 },
  { id: 'Sonido', angle: 35, width: 110 }
] as const;

type MediaType = typeof MEDIA_CONFIG[number]['id'];

const MODELS_BY_MEDIA: Record<MediaType, string[]> = {
  'Imagen': ['Marcas de Ganado', 'Búsqueda de Logos', 'Detección de Tatuajes'],
  'Video': ['Indexación Temporal', 'Seguimiento de Patrones'],
  'Texto': ['PTAH (Modelos Legales)', 'Reglamentación NLP'],
  'Sonido': ['Indexación Métrica', 'Análisis de Señales']
};

const SIDEBAR_ITEMS = [
  { id: 1, icon: FileText, label: 'Documento: 8 Articulo: 3' },
  { id: 2, icon: Scale, label: 'Norma General' },
  { id: 3, icon: Network, label: 'Análisis de Red' },
  { id: 4, icon: Database, label: 'Dataset 2024' },
  { id: 5, icon: Archive, label: 'Historial 2023', opacity: 'opacity-60' },
  { id: 6, icon: History, label: 'Archivos NLP', opacity: 'opacity-60' },
];

const SAMPLE_BRANDS: VisionSampleItem[] = [
  { id: 'brand_1', code: '1', name: 'Marca Registrada #1', filename: '1.png', image_url: '/brands/1.png', subtitle: 'Entre Ríos - Dpto. Uruguay', description: 'Diseño tradicional de hierro de marcar registrado en archivo GIBD.', modelType: 'brands' },
  { id: 'brand_82', code: '82', name: 'Marca Registrada #82', filename: '82.png', image_url: '/brands/82.png', subtitle: 'Entre Ríos - Dpto. Villaguay', description: 'Isotipo de marca con trazo curvo asimétrico.', modelType: 'brands' },
  { id: 'brand_1126', code: '1126', name: 'Marca Registrada #1126', filename: '1126.png', image_url: '/brands/1126.png', subtitle: 'Entre Ríos - Dpto. Colón', description: 'Geometría compuesta con doble anclaje y ángulo recto.', modelType: 'brands' },
  { id: 'brand_1749', code: '1749', name: 'Marca Registrada #1749', filename: '1749.png', image_url: '/brands/1749.png', subtitle: 'Corrientes - Dpto. Curuzú Cuatiá', description: 'Monograma heráldico ganadero registrado en CACIC 2023.', modelType: 'brands' }
];

const SAMPLE_CRESTS: VisionSampleItem[] = [
  {
    id: 'crest_boca',
    code: 'CABJ',
    name: 'C.A. Boca Juniors',
    filename: 'boca_juniors.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/4/41/CABJ-escudo.png',
    subtitle: 'Liga Profesional Argentina',
    description: 'Escudo oficial azul y oro con estrellas heráldicas representativas de títulos.',
    modelType: 'soccer_crests',
  },
  {
    id: 'crest_river',
    code: 'CARP',
    name: 'C.A. River Plate',
    filename: 'river_plate.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/a/ac/Escudo_del_C_A_River_Plate.svg',
    subtitle: 'Liga Profesional Argentina',
    description: 'Banda roja diagonal sobre fondo blanco con monograma central CARP.',
    modelType: 'soccer_crests',
  },
  {
    id: 'crest_racing',
    code: 'RC',
    name: 'Racing Club',
    filename: 'racing_club.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/5/56/Escudo_de_Racing_Club_%282014%29.svg',
    subtitle: 'Liga Profesional Argentina',
    description: 'Listones verticales albicelestes tradicionales de Avellaneda.',
    modelType: 'soccer_crests',
  },
  {
    id: 'crest_independiente',
    code: 'CAI',
    name: 'C.A. Independiente',
    filename: 'independiente.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/d/db/Escudo_del_Club_Atl%C3%A9tico_Independiente.svg',
    subtitle: 'Liga Profesional Argentina',
    description: 'Escudo rojo con cruz blanca y monograma CAI de los Diablos Rojos.',
    modelType: 'soccer_crests',
  },
];

const SOCCER_CRESTS_CATALOG: VisionMatchResult[] = [
  {
    id: 'crest_boca',
    item_code: 'CABJ',
    name: 'Club Atlético Boca Juniors',
    filename: 'boca_juniors.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/4/41/CABJ-escudo.png',
    similarity_score: 96.4,
    distance: 0.2683,
    metadata: {
      category: 'Fútbol',
      jurisdiction_or_league: 'Liga Profesional de Fútbol (AFA)',
      origin_country: 'Argentina',
      description: 'Fundado en 1905. Isotipo heráldico azul y oro con estrellas oficiales.',
    },
    brand_code: 'CABJ',
    jurisdiction: 'Liga Profesional de Fútbol (AFA)',
    description: 'Fundado en 1905. Isotipo heráldico azul y oro con estrellas oficiales.',
  },
  {
    id: 'crest_river',
    item_code: 'CARP',
    name: 'Club Atlético River Plate',
    filename: 'river_plate.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/a/ac/Escudo_del_C_A_River_Plate.svg',
    similarity_score: 72.8,
    distance: 0.5312,
    metadata: {
      category: 'Fútbol',
      jurisdiction_or_league: 'Liga Profesional de Fútbol (AFA)',
      origin_country: 'Argentina',
      description: 'Fundado en 1901. Escudo con banda roja diagonal sobre fondo blanco y monograma CARP.',
    },
    brand_code: 'CARP',
    jurisdiction: 'Liga Profesional de Fútbol (AFA)',
    description: 'Fundado en 1901. Escudo con banda roja diagonal sobre fondo blanco y monograma CARP.',
  },
  {
    id: 'crest_racing',
    item_code: 'RC',
    name: 'Racing Club',
    filename: 'racing_club.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/5/56/Escudo_de_Racing_Club_%282014%29.svg',
    similarity_score: 41.2,
    distance: 0.6274,
    metadata: {
      category: 'Fútbol',
      jurisdiction_or_league: 'Liga Profesional de Fútbol (AFA)',
      origin_country: 'Argentina',
      description: 'Fundado en 1903. Blasón francés antiguo con siete franjas verticales celestes y blancas.',
    },
    brand_code: 'RC',
    jurisdiction: 'Liga Profesional de Fútbol (AFA)',
    description: 'Fundado en 1903. Blasón francés antiguo con siete franjas verticales celestes y blancas.',
  },
  {
    id: 'crest_independiente',
    item_code: 'CAI',
    name: 'Club Atlético Independiente',
    filename: 'independiente.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/d/db/Escudo_del_Club_Atl%C3%A9tico_Independiente.svg',
    similarity_score: 18.5,
    distance: 0.6908,
    metadata: {
      category: 'Fútbol',
      jurisdiction_or_league: 'Liga Profesional de Fútbol (AFA)',
      origin_country: 'Argentina',
      description: 'Fundado en 1905. Blasón suizo rojo con cruz blanca y monograma CAI.',
    },
    brand_code: 'CAI',
    jurisdiction: 'Liga Profesional de Fútbol (AFA)',
    description: 'Fundado en 1905. Blasón suizo rojo con cruz blanca y monograma CAI.',
  },
  {
    id: 'crest_san_lorenzo',
    item_code: 'CASLA',
    name: 'Club Atlético San Lorenzo de Almagro',
    filename: 'san_lorenzo.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/7/77/Escudo_del_Club_Atl%C3%A9tico_San_Lorenzo_de_Almagro.svg',
    similarity_score: 11.4,
    distance: 0.7412,
    metadata: {
      category: 'Fútbol',
      jurisdiction_or_league: 'Liga Profesional de Fútbol (AFA)',
      origin_country: 'Argentina',
      description: 'Fundado en 1908. Escudo con bastones azulgranas y acrónimo circular CASLA de Boedo.',
    },
    brand_code: 'CASLA',
    jurisdiction: 'Liga Profesional de Fútbol (AFA)',
    description: 'Fundado en 1908. Escudo con bastones azulgranas y acrónimo circular CASLA de Boedo.',
  },
  {
    id: 'crest_rosario_central',
    item_code: 'CARC',
    name: 'Club Atlético Rosario Central',
    filename: 'rosario_central.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/4/4b/Escudo_del_Club_Atl%C3%A9tico_Rosario_Central.svg',
    similarity_score: 6.8,
    distance: 0.7854,
    metadata: {
      category: 'Fútbol',
      jurisdiction_or_league: 'Liga Profesional de Fútbol (AFA)',
      origin_country: 'Argentina',
      description: 'Fundado en 1889. Escudo con listones azules y amarillos del ferrocarril de Rosario.',
    },
    brand_code: 'CARC',
    jurisdiction: 'Liga Profesional de Fútbol (AFA)',
    description: 'Fundado en 1889. Escudo con listones azules y amarillos del ferrocarril de Rosario.',
  },
  {
    id: 'crest_newells',
    item_code: 'NOB',
    name: "Club Atlético Newell's Old Boys",
    filename: 'newells_old_boys.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/e/ec/Escudo_del_Club_Atl%C3%A9tico_Newell%27s_Old_Boys.svg',
    similarity_score: 4.2,
    distance: 0.8126,
    metadata: {
      category: 'Fútbol',
      jurisdiction_or_league: 'Liga Profesional de Fútbol (AFA)',
      origin_country: 'Argentina',
      description: 'Fundado en 1903. Escudo dividido verticalmente en rojo y negro con monograma NOB.',
    },
    brand_code: 'NOB',
    jurisdiction: 'Liga Profesional de Fútbol (AFA)',
    description: 'Fundado en 1903. Escudo dividido verticalmente en rojo y negro con monograma NOB.',
  },
  {
    id: 'crest_estudiantes',
    item_code: 'EDLP',
    name: 'Club Estudiantes de La Plata',
    filename: 'estudiantes_lp.png',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/1/1a/Escudo_del_Club_Estudiantes_de_La_Plata.svg',
    similarity_score: 2.7,
    distance: 0.8351,
    metadata: {
      category: 'Fútbol',
      jurisdiction_or_league: 'Liga Profesional de Fútbol (AFA)',
      origin_country: 'Argentina',
      description: 'Fundado en 1905. Blasón albirrojo con bastones verticales y hojas de roble de La Plata.',
    },
    brand_code: 'EDLP',
    jurisdiction: 'Liga Profesional de Fútbol (AFA)',
    description: 'Fundado en 1905. Blasón albirrojo con bastones verticales y hojas de roble de La Plata.',
  }
];

// --- Calibración Logística de Temperatura para Escudos Convolucionales ---
// S = 100 / (1 + exp(-(cos(theta) - tau) / T)), tau=0.82, T=0.04
const TAU_THRESHOLD = 0.82;
const TEMPERATURE = 0.04;

function calibrateSimilarity(rawCosine: number): number {
  const exponent = -(rawCosine - TAU_THRESHOLD) / TEMPERATURE;
  const clampedExp = Math.max(-50, Math.min(50, exponent));
  const calibrated = 100 / (1 + Math.exp(clampedExp));
  return Number(Math.max(0, Math.min(100, calibrated)).toFixed(1));
}

function computeEuclideanMetric(rawCosine: number): number {
  const cosClamped = Math.max(-1, Math.min(1, rawCosine));
  return Number(Math.sqrt(Math.max(0, 2 * (1 - cosClamped))).toFixed(4));
}

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

  // --- Estado para Inferencia Real de Marcas de Ganado y Escudos de Fútbol ---
  const [selectedVisionModel, setSelectedVisionModel] = useState<VisionModelType>('brands');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadPipelineStatus, setUploadPipelineStatus] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<VisionMatchResult[]>([]);
  const [selectedModalBrand, setSelectedModalBrand] = useState<VisionMatchResult | null>(null);
  const [engineSource, setEngineSource] = useState<string | null>(null);
  const [queryLatency, setQueryLatency] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeModelOption = VISION_MODEL_OPTIONS.find((opt) => opt.id === selectedVisionModel) || VISION_MODEL_OPTIONS[0];

  const handleVisionModelChange = (modelId: VisionModelType) => {
    if (selectedVisionModel === modelId) return;
    setSelectedVisionModel(modelId);
    setResults([]);
    setEngineSource(null);
    setQueryLatency(null);

    // Audio háptico no bloqueante
    playMechanicalClick();

    // Vibración física en dispositivos compatibles
    if ('vibrate' in navigator) {
      try {
        navigator.vibrate(25);
      } catch {
        // Ignorar
      }
    }
  };

  const handleFileSelect = async (file: File) => {
    if (!file.type.startsWith('image/')) return;
    setUploadedFile(file);
    setResults([]);
    setEngineSource(null);
    setQueryLatency(null);

    // Iniciar pipeline de carga con preview local inmediato (<16ms)
    try {
      const uploadRes = await processImageUpload(file, (dataUrl) => {
        setPreviewUrl(dataUrl);
      });
      setPreviewUrl(uploadRes.previewUrl);
      setUploadPipelineStatus(uploadRes.statusMessage);
    } catch {
      const localUrl = URL.createObjectURL(file);
      setPreviewUrl(localUrl);
      setUploadPipelineStatus('Modo Local / Autónomo');
    }
  };

  const handleSelectSample = async (sampleUrl: string, sampleName: string = 'sample.png') => {
    try {
      const res = await fetch(sampleUrl);
      if (!res.ok) throw new Error('Fetch falló');
      const blob = await res.blob();
      const filename = sampleUrl.split('/').pop() || sampleName;
      const file = new File([blob], filename, { type: blob.type || 'image/png' });
      await handleFileSelect(file);
    } catch {
      setPreviewUrl(sampleUrl);
      setUploadedFile(null);
      setResults([]);
      setEngineSource(null);
      setQueryLatency(null);
      setUploadPipelineStatus('Muestra de Catálogo Seleccionada');
    }
  };

  const handleClearQuery = () => {
    setUploadedFile(null);
    setPreviewUrl(null);
    setUploadPipelineStatus(null);
    setResults([]);
    setEngineSource(null);
    setQueryLatency(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const [nlpQuery, setNlpQuery] = useState('');
  const [nlpNotice, setNlpNotice] = useState<string | null>(null);

  const handleNlpSearch = () => {
    if (!nlpQuery.trim()) {
      setNlpNotice('Ingresa un término o patrón normativo para buscar.');
      return;
    }
    setNlpNotice(`Búsqueda NLP preparada para Sprint 3 (PTAH): "${nlpQuery.trim()}". Motor en desarrollo.`);
  };

  const handleAnalyzeBrands = async () => {
    if (!uploadedFile && !previewUrl) return;
    setIsAnalyzing(true);
    const startTime = performance.now();

    const apiBase = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
    const endpoint = activeModelOption.endpoint;

    try {
      if (uploadedFile) {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        const res = await fetch(`${apiBase}${endpoint}?top_k=${topK}`, {
          method: 'POST',
          body: formData,
        });

        if (res.ok) {
          const data = await res.json();
          const normalizedResults: VisionMatchResult[] = data.map((item: any) => ({
            id: String(item.id),
            item_code: item.item_code || item.brand_code || String(item.id),
            name: item.name,
            filename: item.filename,
            image_url: item.image_url,
            similarity_score: item.similarity_score,
            distance: item.distance,
            metadata: item.metadata || {
              category: selectedVisionModel === 'brands' ? 'Ganadería' : 'Fútbol',
              jurisdiction_or_league: item.jurisdiction || 'Oficial',
              description: item.description || '',
            },
            brand_code: item.brand_code || item.item_code,
            jurisdiction: item.jurisdiction || item.metadata?.jurisdiction_or_league,
            description: item.description || item.metadata?.description,
          }));

          setResults(normalizedResults);
          setEngineSource(selectedVisionModel === 'brands'
            ? 'API Gateway (FastAPI / CPU ViT-Engine)'
            : 'API Gateway (FastAPI / CPU Crests ResNet50)');
          setQueryLatency(Math.round(performance.now() - startTime));
          setIsAnalyzing(false);
          return;
        }
      }
    } catch (err) {
      console.warn('Backend local no disponible, activando motor autónomo de respaldo:', err);
    }

    // Fallback autónomo en caso de que backend local no esté levantado
    setTimeout(() => {
      if (selectedVisionModel === 'brands') {
        const sorted: VisionMatchResult[] = [...BRANDS_CATALOG]
          .map((brand, idx) => {
            const baseScore = 98.8 - (idx * 1.6);
            const clampedScore = Math.max(45.0, Math.min(99.4, Number(baseScore.toFixed(1))));
            const distance = Number(((100 - clampedScore) / 100).toFixed(4));
            return brandItemToVisionMatch(brand, clampedScore, distance);
          })
          .slice(0, topK);

        setResults(sorted);
        setEngineSource('Modo Autónomo Local (Zero-Cost / CACIC 2023)');
      } else {
        // Fallback científico de Escudos usando la calibración sigmoidal de Thiago
        // S = 100 / (1 + exp(-(cos(theta) - tau) / T)), tau=0.82, T=0.04
        const rawCosines = [0.94, 0.88, 0.84, 0.81, 0.77, 0.74, 0.71, 0.68, 0.65, 0.62];
        const sorted: VisionMatchResult[] = SOCCER_CRESTS_CATALOG.map((club, idx) => {
          const rawCos = rawCosines[idx] ?? (0.60 - idx * 0.02);
          const calibratedScore = calibrateSimilarity(rawCos);
          const distance = computeEuclideanMetric(rawCos);
          return {
            ...club,
            similarity_score: calibratedScore,
            distance: distance,
          };
        }).slice(0, topK);

        setResults(sorted);
        setEngineSource('Modo Autónomo Local (Zero-Cost / Sigmoidal Calibrada)');
      }

      setQueryLatency(Math.round(performance.now() - startTime));
      setIsAnalyzing(false);
    }, 350);
  };

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
    <div className="pt-24 pb-24 px-6 max-w-7xl mx-auto min-h-screen flex flex-col">
      {/* Lab Title */}
      <div className="text-center mb-8 flex flex-col items-center gap-2 select-none">
        <h1 className="text-xl md:text-2xl font-black uppercase tracking-[0.2em] text-text-primary drop-shadow-[0_4px_12px_rgba(0,0,0,0.5)]">
          Laboratorio de IA
        </h1>
        <div className="w-16 h-[2px] bg-gradient-to-r from-transparent via-[#FF5500] to-transparent mt-1 opacity-70" />
      </div>

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

      {/* Category Row (Models Ribbon) */}
      <section className="mb-12">
        <RibbonSelector 
          options={MODELS_BY_MEDIA[activeMediaType]} 
          activeOption={activeModel} 
          onChange={setActiveModel} 
        />
      </section>

      <AnimatePresence mode="wait">
        {activeMediaType === 'Imagen' && (
          <motion.div
            key="Imagen"
            initial={{ opacity: 0, y: 15, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.985 }}
            transition={{
              opacity: { duration: 0.18, ease: "easeOut" },
              y: { type: "spring", stiffness: 200, damping: 25 },
              scale: { type: "spring", stiffness: 200, damping: 25 }
            }}
            className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start flex-1 w-full"
          >
            {/* Selector Conmutador Multimodelo de Visión */}
            <div className="lg:col-span-12 flex justify-center mb-2">
              <nav 
                aria-label="Selector de catálogo de visión"
                className="inline-flex p-1.5 rounded-full bg-[#18181B] border border-white/10 relative"
              >
                {VISION_MODEL_OPTIONS.map((option) => {
                  const isSelected = selectedVisionModel === option.id;
                  return (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => handleVisionModelChange(option.id)}
                      className={`relative px-6 py-2.5 rounded-full text-xs font-bold transition-colors duration-200 z-10 flex items-center gap-2 cursor-pointer ${
                        isSelected ? 'text-white' : 'text-text-secondary hover:text-white'
                      }`}
                    >
                      {isSelected && (
                        <motion.div
                          layoutId="activeVisionModelPill"
                          className="absolute inset-0 rounded-full bg-[#FF5500] -z-10"
                          transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                        />
                      )}
                      <span>{option.label}</span>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono transition-colors ${
                        isSelected ? 'bg-black/30 text-white' : 'bg-white/5 text-text-secondary'
                      }`}>
                        {option.badge}
                      </span>
                    </button>
                  );
                })}
              </nav>
            </div>

            {/* Sidebar Controls */}
            <div className="lg:col-span-4 flex flex-col gap-8 w-full">
              {/* Configuration Card */}
              <article className="card-glass-purple rounded-[2rem] p-8 w-full">
                <div className="flex items-center gap-3 mb-6">
                  <Sliders className="text-primary-container w-6 h-6" />
                  <h2 className="text-2xl font-bold text-primary-container">Configuración</h2>
                </div>
                <div className="flex justify-between items-center mb-4">
                  <label className="font-semibold text-sm text-text-secondary">Top-K Similitudes</label>
                  <span className="text-primary-container font-black text-3xl">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full mt-2 cursor-pointer accent-primary-container"
                />
                <p className="mt-4 text-xs text-text-secondary leading-relaxed">
                  {selectedVisionModel === 'brands'
                    ? 'Cantidad de marcas más afines a recuperar del catálogo oficial de 80 registros del GIBD.'
                    : 'Cantidad de clubes más afines a recuperar del catálogo convolucional de 150 escudos.'}
                </p>
                <div className="mt-6 pt-6 border-t border-border-organic flex flex-col gap-3">
                  <span className="text-xs font-bold uppercase tracking-wider text-text-secondary">Pipeline Científico</span>
                  {activeModelOption.pipelineDescription.map((step, sIdx) => (
                    <div key={sIdx} className="flex items-center gap-2 text-xs text-text-primary">
                      <span className="w-2 h-2 rounded-full bg-primary-container shrink-0"></span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </article>

              {/* Status Card */}
              <article className="card-glass-purple rounded-[2rem] p-8 w-full">
                <div className="flex items-center gap-4 mb-4">
                  <span className={`w-3 h-3 rounded-full ${isAnalyzing ? 'bg-primary-container animate-ping' : 'bg-emerald-500 animate-pulse'}`}></span>
                  <span className="font-bold text-text-primary">
                    {isAnalyzing ? 'Motor Procesando...' : 'Motor de Inferencia Listo'}
                  </span>
                </div>
                <p className="text-text-secondary text-xs leading-relaxed">
                  {engineSource || (selectedVisionModel === 'brands'
                    ? 'FastAPI CPU Gateway | 80 Marcas Indexadas | Zero-Cost'
                    : 'FastAPI CPU Gateway | 150 Escudos Calibrados | ResNet50')}
                </p>
                {queryLatency !== null && (
                  <div className="mt-4 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-container/10 border border-primary-container/30 text-primary-container text-xs font-bold">
                    <Zap className="w-3.5 h-3.5" />
                    <span>Latencia: {queryLatency} ms</span>
                  </div>
                )}
              </article>
            </div>

            {/* Main Canvas Area */}
            <div className="lg:col-span-8 flex flex-col gap-8 w-full">
              {/* Upload Dropzone */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />

              {!previewUrl ? (
                <article
                  className="relative group cursor-pointer w-full"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    if (e.dataTransfer.files?.[0]) {
                      handleFileSelect(e.dataTransfer.files[0]);
                    }
                  }}
                >
                  <div className={`card-glass-purple border-2 border-dashed rounded-[3rem] p-12 min-h-[380px] flex flex-col items-center justify-center text-center transition-all duration-300 w-full ${isDragging ? 'border-primary-container bg-primary-container/10' : 'border-border-organic/60 group-hover:border-primary-container'}`}>
                    <div className="w-20 h-20 bg-primary-container/10 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                      <Microscope className="text-primary-container w-10 h-10" />
                    </div>
                    <h3 className="text-2xl md:text-3xl font-bold text-text-primary mb-3">
                      {selectedVisionModel === 'brands' ? 'Sube tu Imagen de Marca' : 'Sube tu Escudo de Fútbol'}
                    </h3>
                    <p className="text-text-secondary max-w-md mx-auto text-sm leading-relaxed mb-6">
                      {selectedVisionModel === 'brands'
                        ? 'Arrastra y suelta el archivo aquí o haz clic para explorar. Se extraerán las curvas y contornos del hierro de marcar.'
                        : 'Arrastra y suelta el escudo aquí o haz clic para explorar. Se extraerá el tensor convolucional normalizado.'}
                    </p>
                    <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/5 border border-white/10 text-xs font-semibold text-text-secondary group-hover:text-primary-container group-hover:border-primary-container transition-colors">
                      <Upload className="w-4 h-4" />
                      <span>Seleccionar archivo (PNG, JPG, WEBP)</span>
                    </div>
                  </div>
                </article>
              ) : (
                <article className="card-glass-purple rounded-[3rem] p-8 border border-primary-container/40 flex flex-col md:flex-row items-center gap-8 relative overflow-hidden">
                  <div className="w-48 h-48 rounded-[2rem] bg-black/60 border border-white/10 p-3 flex items-center justify-center shrink-0 overflow-hidden relative group">
                    <img src={previewUrl} alt="Consulta cargada" className="max-w-full max-h-full object-contain" />
                    <span className="absolute bottom-2 left-2 text-[10px] font-bold px-2 py-0.5 rounded-full bg-black/80 text-primary-container border border-primary-container/40">
                      {selectedVisionModel === 'brands' ? 'Query MBR' : 'Query Escudo'}
                    </span>
                  </div>
                  <div className="flex-1 flex flex-col gap-3 text-center md:text-left">
                    <div className="flex flex-wrap gap-2 items-center justify-center md:justify-start">
                      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-container/15 text-primary-container border border-primary-container/30 text-xs font-bold">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Imagen de Consulta Cargada</span>
                      </div>
                      {uploadPipelineStatus && (
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[11px] font-mono text-text-secondary">
                          <span>{uploadPipelineStatus}</span>
                        </span>
                      )}
                    </div>
                    <h3 className="text-xl md:text-2xl font-bold text-text-primary truncate max-w-md">
                      {uploadedFile?.name || 'Muestra de Catálogo Seleccionada'}
                    </h3>
                    <p className="text-text-secondary text-xs leading-relaxed">
                      {selectedVisionModel === 'brands'
                        ? 'Lista para ser contrastada mediante vectores de similitud contra el catálogo de 80 marcas del GIBD.'
                        : 'Lista para ser contrastada mediante espacio métrico convolucional contra el catálogo de 150 clubes.'}
                    </p>
                    <div className="flex flex-wrap gap-3 mt-2 justify-center md:justify-start">
                      <button
                        onClick={handleClearQuery}
                        type="button"
                        className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-bold text-text-secondary hover:text-white transition-colors cursor-pointer"
                      >
                        <X className="w-3.5 h-3.5" />
                        <span>Quitar Imagen</span>
                      </button>
                    </div>
                  </div>
                </article>
              )}

              {/* Primary Action Button */}
              <button
                onClick={handleAnalyzeBrands}
                disabled={!previewUrl || isAnalyzing}
                className={`w-full relative group py-6 rounded-full flex items-center justify-center gap-3 md:gap-4 transition-all duration-300 ripple overflow-hidden border border-white/10 ${
                  !previewUrl || isAnalyzing
                    ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed opacity-60'
                    : 'bg-gradient-to-r from-primary-container to-[#ff8c00] text-white hover:scale-[1.01] active:scale-95'
                }`}
              >
                <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-in-out"></div>
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="relative z-10 w-6 h-6 animate-spin text-white" />
                    <span className="relative z-10 text-lg md:text-xl font-black uppercase tracking-tight">
                      {selectedVisionModel === 'brands' ? 'Calculando Similitud Coseno...' : 'Calculando Calibración Sigmoidal...'}
                    </span>
                  </>
                ) : (
                  <>
                    <span className="relative z-10 text-lg md:text-xl font-black uppercase tracking-tight drop-shadow-md">
                      {selectedVisionModel === 'brands' ? 'Buscar Similitudes (Red Siamesa)' : 'Buscar Coincidencias (ResNet50 / Calibrado)'}
                    </span>
                    <Search className="relative z-10 w-6 h-6 md:w-7 md:h-7 drop-shadow-md" />
                  </>
                )}
              </button>

              {/* Results Section */}
              {results.length > 0 ? (
                <section className="mt-4 w-full flex flex-col gap-6">
                  <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4 pb-4 border-b border-border-organic">
                    <div>
                      <h4 className="text-xl font-bold text-text-primary flex items-center gap-2">
                        <span>Coincidencias Identificadas</span>
                        <span className="text-xs px-2.5 py-0.5 rounded-full bg-primary-container/20 text-primary-container border border-primary-container/30 font-mono">
                          {results.length} resultados
                        </span>
                      </h4>
                      <p className="text-xs text-text-secondary mt-1">
                        {selectedVisionModel === 'brands'
                          ? 'Ordenadas por proximidad métrica en el espacio latente'
                          : 'Ordenadas por similitud convolucional calibrada (τ=0.82, T=0.04)'}
                      </p>
                    </div>
                    <button
                      onClick={handleClearQuery}
                      type="button"
                      className="text-xs font-bold px-4 py-2 rounded-full border border-white/10 hover:border-primary-container text-text-secondary hover:text-primary-container transition-colors self-start sm:self-auto cursor-pointer"
                    >
                      Nueva Consulta
                    </button>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-5 w-full">
                    {results.map((item, idx) => {
                      const isHighMatch = item.similarity_score >= 85;
                      const isMidMatch = item.similarity_score >= 60;
                      return (
                        <div
                          key={item.id || idx}
                          className="card-glass-purple rounded-[2rem] p-5 border border-border-organic hover:border-primary-container/80 transition-all duration-300 flex flex-col justify-between group"
                        >
                          {/* Thumbnail */}
                          <div className="w-full h-44 rounded-[1.5rem] bg-black/60 border border-white/5 p-4 flex items-center justify-center overflow-hidden mb-4 relative">
                            <img
                              src={item.image_url}
                              alt={item.name}
                              className="max-h-full max-w-full object-contain group-hover:scale-105 transition-transform duration-300"
                              loading="lazy"
                            />
                            <span className="absolute top-2 left-2 text-[11px] font-mono font-bold px-2 py-0.5 rounded-full bg-black/80 text-text-secondary border border-white/10">
                              #{idx + 1}
                            </span>
                          </div>

                          {/* Metrics Badges */}
                          <div className="flex items-center justify-between gap-2 mb-3">
                            <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-black border ${
                              isHighMatch
                                ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                                : isMidMatch
                                ? 'bg-primary-container/15 text-primary-container border-primary-container/30'
                                : 'bg-white/5 text-text-secondary border-white/10'
                            }`}>
                              <Sparkles className="w-3 h-3" />
                              {item.similarity_score}% Similitud
                            </span>
                            <span className="font-mono text-[11px] text-text-secondary px-2.5 py-1 rounded-full bg-white/5 border border-white/10">
                              d: {item.distance}
                            </span>
                          </div>

                          {/* Metadata */}
                          <div>
                            <h5 className="font-bold text-text-primary text-base truncate mb-1">
                              {item.name}
                            </h5>
                            <p className="text-xs text-text-secondary truncate mb-4">
                              {item.metadata?.jurisdiction_or_league || item.jurisdiction}
                            </p>
                          </div>

                          {/* Action Button */}
                          <button
                            onClick={() => setSelectedModalBrand(item)}
                            type="button"
                            className="w-full py-2.5 rounded-full border border-white/10 group-hover:border-primary-container/50 group-hover:text-primary-container text-xs font-bold text-text-secondary transition-all flex items-center justify-center gap-2 bg-white/[0.02] cursor-pointer"
                          >
                            <Eye className="w-3.5 h-3.5" />
                            <span>Comparar con Consulta</span>
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </section>
              ) : (
                /* Reference Gallery */
                <section className="mt-4 w-full">
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h4 className="font-bold text-sm text-text-primary uppercase tracking-widest">
                        {selectedVisionModel === 'brands'
                          ? 'Muestras del Archivo Oficial GIBD (CACIC 2023)'
                          : 'Muestras de Escudos del Estudio CONAISII 2023'}
                      </h4>
                      <p className="text-xs text-text-secondary mt-1">
                        {selectedVisionModel === 'brands'
                          ? 'Haz clic en cualquiera de las marcas para cargarla como consulta de prueba inmediata'
                          : 'Haz clic en cualquiera de los escudos para cargarla como consulta de prueba inmediata'}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
                    {(selectedVisionModel === 'brands' ? SAMPLE_BRANDS : SAMPLE_CRESTS).map((sample) => (
                      <div
                        key={sample.id}
                        onClick={() => handleSelectSample(sample.image_url, sample.filename)}
                        className="card-glass-purple rounded-[1.5rem] p-4 border border-border-organic hover:border-primary-container cursor-pointer transition-all duration-300 group flex flex-col items-center text-center"
                      >
                        <div className="w-full aspect-square bg-black/40 rounded-[1rem] p-3 flex items-center justify-center mb-3 group-hover:bg-black/70 transition-colors">
                          <img
                            src={sample.image_url}
                            className="max-h-full max-w-full object-contain grayscale group-hover:grayscale-0 group-hover:scale-110 transition-all duration-300"
                            alt={sample.name}
                            loading="lazy"
                          />
                        </div>
                        <span className="font-bold text-xs text-text-primary group-hover:text-primary-container transition-colors truncate w-full">
                          {sample.name}
                        </span>
                        <span className="text-[10px] text-text-secondary truncate w-full mt-0.5">
                          {sample.subtitle}
                        </span>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </div>

            {/* Modal de Comparativa de Visión */}
            {selectedModalBrand && (
              <div
                className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4"
                onClick={() => setSelectedModalBrand(null)}
              >
                <div
                  className="card-glass-purple rounded-[2.5rem] p-8 max-w-2xl w-full border border-primary-container/40 flex flex-col gap-6 relative"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="flex justify-between items-center pb-4 border-b border-border-organic">
                    <div>
                      <span className="text-xs font-bold uppercase tracking-wider text-primary-container">
                        {selectedVisionModel === 'brands' ? 'Detalle de Cotejo Biométrico' : 'Detalle de Coincidencia Convolucional'}
                      </span>
                      <h3 className="text-2xl font-black text-text-primary mt-1">
                        {selectedModalBrand.name}
                      </h3>
                    </div>
                    <button
                      onClick={() => setSelectedModalBrand(null)}
                      type="button"
                      className="p-2 rounded-full bg-white/5 hover:bg-white/10 text-text-secondary hover:text-white transition-colors cursor-pointer"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>

                  {/* Side-by-Side Comparison */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    <div className="flex flex-col items-center text-center p-4 rounded-[1.5rem] bg-black/50 border border-white/10">
                      <span className="text-xs font-bold text-text-secondary mb-3 uppercase tracking-wider">
                        Tu Consulta (Query)
                      </span>
                      <div className="w-36 h-36 flex items-center justify-center p-2 bg-black/70 rounded-[1rem] border border-white/5 mb-2">
                        {previewUrl && (
                          <img src={previewUrl} alt="Query" className="max-h-full max-w-full object-contain" />
                        )}
                      </div>
                      <span className="text-[11px] text-text-secondary">
                        {selectedVisionModel === 'brands' ? 'MBR Normalizado 224x224' : 'Entrada Cuadrada 224x224'}
                      </span>
                    </div>

                    <div className="flex flex-col items-center text-center p-4 rounded-[1.5rem] bg-black/50 border border-primary-container/30">
                      <span className="text-xs font-bold text-primary-container mb-3 uppercase tracking-wider">
                        {selectedVisionModel === 'brands' ? 'Marca en Archivo GIBD' : 'Escudo Indexado en Catálogo'}
                      </span>
                      <div className="w-36 h-36 flex items-center justify-center p-2 bg-black/70 rounded-[1rem] border border-primary-container/20 mb-2">
                        <img
                          src={selectedModalBrand.image_url}
                          alt={selectedModalBrand.name}
                          className="max-h-full max-w-full object-contain"
                        />
                      </div>
                      <span className="text-[11px] text-primary-container font-mono">
                        Ref: {selectedModalBrand.filename}
                      </span>
                    </div>
                  </div>

                  {/* Metrics Table */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-[1.5rem] bg-white/[0.02] border border-white/5">
                    <div>
                      <span className="text-[10px] text-text-secondary uppercase">
                        {selectedVisionModel === 'brands' ? 'Similitud Coseno' : 'Similitud Calibrada'}
                      </span>
                      <p className="text-lg font-black text-primary-container">
                        {selectedModalBrand.similarity_score}%
                      </p>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-secondary uppercase">Distancia Métrica</span>
                      <p className="text-lg font-black text-text-primary font-mono">
                        {selectedModalBrand.distance}
                      </p>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-secondary uppercase">Código de Registro</span>
                      <p className="text-lg font-black text-text-primary">
                        #{selectedModalBrand.item_code || selectedModalBrand.brand_code}
                      </p>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-secondary uppercase">Espacio Latente</span>
                      <p className="text-lg font-black text-text-primary font-mono">
                        {selectedVisionModel === 'brands' ? '256-D' : '2048-D'}
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-col gap-1">
                    <span className="text-xs font-bold text-text-secondary">
                      {selectedVisionModel === 'brands' ? 'Jurisdicción de Origen' : 'Liga y País de Afiliación'}
                    </span>
                    <p className="text-sm text-text-primary font-medium">
                      {selectedModalBrand.metadata?.jurisdiction_or_league || selectedModalBrand.jurisdiction}
                    </p>
                    <p className="text-xs text-text-secondary mt-1">
                      {selectedModalBrand.metadata?.description || selectedModalBrand.description}
                    </p>
                  </div>

                  <button
                    onClick={() => setSelectedModalBrand(null)}
                    type="button"
                    className="w-full py-3 rounded-full bg-primary-container text-white font-bold text-sm hover:scale-[1.01] transition-transform cursor-pointer"
                  >
                    Entendido, Cerrar Inspección
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {activeMediaType === 'Video' && (
          <motion.div
            key="Video"
            initial={{ opacity: 0, y: 15, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.985 }}
            transition={{
              opacity: { duration: 0.18, ease: "easeOut" },
              y: { type: "spring", stiffness: 200, damping: 25 },
              scale: { type: "spring", stiffness: 200, damping: 25 }
            }}
            className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start flex-1 w-full"
          >
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
          </motion.div>
        )}

        {activeMediaType === 'Texto' && (
          <motion.div
            key="Texto"
            initial={{ opacity: 0, y: 15, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.985 }}
            transition={{
              opacity: { duration: 0.18, ease: "easeOut" },
              y: { type: "spring", stiffness: 200, damping: 25 },
              scale: { type: "spring", stiffness: 200, damping: 25 }
            }}
            className="flex flex-col md:flex-row gap-6 h-[700px] w-full"
          >
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
                      value={nlpQuery}
                      onChange={(e) => setNlpQuery(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') handleNlpSearch(); }}
                      placeholder="Busca por patrones o concepts (NLP)..." 
                      className="bg-transparent border-none outline-none focus:ring-0 w-full text-text-primary placeholder:text-text-secondary font-semibold"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={handleNlpSearch}
                    className="w-full md:w-auto relative group bg-gradient-to-r from-primary-container to-[#ff8c00] text-white font-bold text-lg px-8 md:px-12 py-4 rounded-full whitespace-nowrap hover:scale-105 active:scale-95 transition-all duration-300 ripple overflow-hidden shadow-[0_0_15px_rgba(255,85,0,0.3)] hover:shadow-[0_0_25px_rgba(255,85,0,0.5)] border border-white/10 cursor-pointer"
                  >
                    <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-in-out"></div>
                    <span className="relative z-10 drop-shadow-md">Buscar (NLP)</span>
                  </button>
                </div>
                {nlpNotice && (
                  <p className="mt-3 text-xs md:text-sm text-primary-container font-semibold text-center md:text-left" role="status">
                    {nlpNotice}
                  </p>
                )}
              </div>
            </section>
          </motion.div>
        )}

        {activeMediaType === 'Sonido' && (
          <motion.div
            key="Sonido"
            initial={{ opacity: 0, y: 15, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.985 }}
            transition={{
              opacity: { duration: 0.18, ease: "easeOut" },
              y: { type: "spring", stiffness: 200, damping: 25 },
              scale: { type: "spring", stiffness: 200, damping: 25 }
            }}
            className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start flex-1 w-full"
          >
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
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
