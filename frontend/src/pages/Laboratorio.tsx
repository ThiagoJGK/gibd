import React, { useState } from 'react';
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

export function Laboratorio() {
  const [activeMediaType, setActiveMediaType] = useState<MediaType>('Imagen');
  const [activeModel, setActiveModel] = useState<string>(MODELS_BY_MEDIA['Imagen'][0]);
  
  const handleMediaTypeChange = (media: MediaType) => {
    setActiveMediaType(media);
    setActiveModel(MODELS_BY_MEDIA[media][0]);
  };
  
  const [topK, setTopK] = useState(10);
  const [isDragging, setIsDragging] = useState(false);
  const [activeSearchItem, setActiveSearchItem] = useState<number | null>(null);

  const activeConfig = MEDIA_CONFIG.find(m => m.id === activeMediaType) || MEDIA_CONFIG[0];

  return (
    <div className="pt-32 pb-24 px-6 max-w-7xl mx-auto min-h-screen flex flex-col">
      {/* Pivot Dial Selector */}
      <div className="relative w-full max-w-[500px] h-[250px] mx-auto mb-16 mt-4 flex justify-center overflow-visible">
        {/* Pivot Base */}
        <div className="absolute bottom-0 w-14 h-14 rounded-full bg-[#1A1124]/95 border border-[#3A234A] shadow-[0_10px_30px_rgba(0,0,0,0.6)] flex items-center justify-center z-30">
          <div className="w-5 h-5 rounded-full bg-white/10 shadow-inner border border-white/5"></div>
        </div>

        {/* Texts on Arc (Background) */}
        {MEDIA_CONFIG.map((media) => {
          const isActive = activeMediaType === media.id;
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
          animate={{ rotate: activeConfig.angle }}
          transition={{ type: "spring", stiffness: 220, damping: 20 }}
          className="absolute bottom-[28px] origin-bottom flex flex-col items-center justify-start z-20 pointer-events-none"
          style={{ height: 200, width: 60 }} // height is the radius
        >
          {/* Pill Head (The Hole) */}
          <motion.div
            initial={false}
            animate={{ width: activeConfig.width }}
            transition={{ type: "spring", stiffness: 220, damping: 20 }}
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
            transition={{ type: "spring", stiffness: 220, damping: 20 }}
            className="flex-1 -mt-[2px] w-full bg-[#1A1124]/95 backdrop-blur-md relative z-10"
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
                  : 'bg-surface-deep text-text-primary border-border-organic hover:bg-secondary-container'
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
            <article className="bg-surface-deep rounded-[2rem] p-8 border border-border-organic w-full">
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
            <article className="bg-surface-deep rounded-[2rem] p-8 border border-border-organic w-full">
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
              <div className={`bg-surface-deep border-2 border-dashed rounded-[3rem] p-12 min-h-[400px] flex flex-col items-center justify-center text-center transition-all duration-300 w-full ${isDragging ? 'border-primary-container bg-secondary-container' : 'border-border-organic group-hover:border-primary-container'}`}>
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
                  <div key={i} className="aspect-square bg-surface-deep rounded-[1.5rem] border border-border-organic overflow-hidden group">
                    <img src={src} className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500" alt={`Reference ${i+1}`} />
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
            <article className="bg-surface-deep rounded-[2rem] p-8 border border-border-organic w-full">
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
                  <div className="w-10 h-6 bg-surface-deep border border-border-organic rounded-full relative">
                    <div className="w-4 h-4 bg-text-secondary rounded-full absolute left-1 top-1"></div>
                  </div>
                </div>
              </div>
            </article>

            {/* Status Card */}
            <article className="bg-surface-deep rounded-[2rem] p-8 border border-border-organic w-full">
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
              <div className={`bg-surface-deep border-2 border-dashed rounded-[3rem] p-12 min-h-[400px] flex flex-col items-center justify-center text-center transition-all duration-300 w-full ${isDragging ? 'border-primary-container bg-secondary-container' : 'border-border-organic group-hover:border-primary-container'}`}>
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
                      : 'bg-surface-deep border-transparent hover:border-primary-container'
                  }`}
                >
                  <item.icon className="text-primary-container w-6 h-6 shrink-0" />
                  <span className="font-semibold text-sm truncate">{item.label}</span>
                </div>
              ))}
            </div>
          </aside>

          {/* Main Content Area */}
          <section className="flex-1 flex flex-col bg-surface-deep rounded-[3rem] overflow-hidden border border-border-organic relative min-h-[500px] md:min-h-0 w-full">
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
            <article className="bg-surface-deep rounded-[2rem] p-8 border border-border-organic w-full">
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
            <article className="bg-surface-deep rounded-[2rem] p-8 border border-border-organic w-full">
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
              <div className={`bg-surface-deep border-2 border-dashed rounded-[3rem] p-12 min-h-[400px] flex flex-col items-center justify-center text-center transition-all duration-300 w-full ${isDragging ? 'border-primary-container bg-secondary-container' : 'border-border-organic group-hover:border-primary-container'}`}>
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
