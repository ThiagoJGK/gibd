import { supabase } from './supabaseClient';

// Semilla inicial para noticias
const MOCK_NOTICIAS_SEED = [
  {
    id: 'noticia-1',
    title: 'Presentación en CoNaIISI 2024 de la investigación en Recuperación de Información de Reglamentación Académica en Español utilizando NLP.',
    content: 'Se presentó el trabajo de investigación desarrollado por el equipo sobre Recuperación de Información de Reglamentación Académica en Español utilizando técnicas avanzadas de Procesamiento del Lenguaje Natural.',
    tag: 'Publicación',
    date: 'Nov 2024',
    image_url: null
  },
  {
    id: 'noticia-2',
    title: 'Aceptación y publicación en ARGENCON 2024 del paper "Advanced Variable Tuning and Biases in Chatbot Models: Analysis of the PTAH Prototype".',
    content: 'El paper del prototipo conversacional PTAH fue publicado en ARGENCON 2024. Este artículo presenta un análisis exhaustivo sobre el sesgo y ajuste de variables en modelos de chatbot.',
    tag: 'Investigación',
    date: 'Oct 2024',
    image_url: null
  },
  {
    id: 'noticia-3',
    title: 'Exposición en CACIC 2024: "Image Feature Extraction for Similarity Searching Using Transfer Learning with ResNet".',
    content: 'El equipo expuso el trabajo que detalla el uso de Transfer Learning con ResNet para la extracción de características visuales en búsquedas por similitud.',
    tag: 'Conferencia',
    date: 'Oct 2024',
    image_url: null
  }
];

/**
 * Sube un archivo al almacenamiento de Supabase o simula la subida en modo local.
 */
export const uploadToStorage = async (file: File, bucket: 'noticias' | 'papers'): Promise<string> => {
  const isMock = !import.meta.env.VITE_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL === 'YOUR_SUPABASE_URL';
  
  if (isMock) {
    return `https://supabase-mock-storage.gibd.utn.edu.ar/${bucket}/${Date.now()}_${file.name}`;
  }

  const fileExt = file.name.split('.').pop();
  const fileName = `${Math.random().toString(36).substring(2)}_${Date.now()}.${fileExt}`;
  const filePath = `${fileName}`;

  const { error: uploadError } = await supabase.storage
    .from(bucket)
    .upload(filePath, file);

  if (uploadError) {
    throw new Error(`Error en Storage: ${uploadError.message}`);
  }

  const { data } = supabase.storage
    .from(bucket)
    .getPublicUrl(filePath);

  return data.publicUrl;
};

// ==========================================
// MÉTODOS PARA DESARROLLADOR A (MÓDULO NOTICIAS)
// ==========================================

/**
 * Obtiene las noticias de la base de datos o LocalStorage si está en modo mock.
 */
export const getNoticias = async (): Promise<any[]> => {
  const isMock = !import.meta.env.VITE_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL === 'YOUR_SUPABASE_URL';
  
  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_noticias');
    if (localData) return JSON.parse(localData);
    
    // Sembrar la semilla inicial
    localStorage.setItem('gibd_mock_noticias', JSON.stringify(MOCK_NOTICIAS_SEED));
    return MOCK_NOTICIAS_SEED;
  }
  
  // Consulta Supabase Real
  const { data, error } = await supabase
    .from('noticias')
    .select('*')
    .order('created_at', { ascending: false });
    
  if (error) throw error;
  return data || [];
};

/**
 * Guarda una nueva noticia en Supabase o en LocalStorage si está en modo mock.
 */
export const saveNoticia = async (noticiaForm: any, noticiaFile: File | null): Promise<void> => {
  const isMock = !import.meta.env.VITE_SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL === 'YOUR_SUPABASE_URL';
  
  let imageUrl = noticiaForm.imageUrl;
  if (noticiaFile) {
    imageUrl = await uploadToStorage(noticiaFile, 'noticias');
  }
  
  const formattedDate = noticiaForm.date || new Date().toLocaleDateString('es-ES', { month: 'short', year: 'numeric' });

  if (isMock) {
    const current = await getNoticias();
    const newNoticia = {
      id: Math.random().toString(36).substring(2),
      title: noticiaForm.title,
      content: noticiaForm.content,
      tag: noticiaForm.tag,
      date: formattedDate,
      image_url: imageUrl || null
    };
    localStorage.setItem('gibd_mock_noticias', JSON.stringify([newNoticia, ...current]));
  } else {
    const { error } = await supabase.from('noticias').insert({
      title: noticiaForm.title,
      content: noticiaForm.content,
      tag: noticiaForm.tag,
      date: formattedDate,
      image_url: imageUrl || null
    });
    if (error) throw error;
  }
};


// ==========================================
// MÉTODOS PARA DESARROLLADOR B (MÓDULO PAPERS E INVESTIGADORES)
// ==========================================

/**
 * Obtiene la lista de investigadores de la base de datos o LocalStorage si está en modo mock.
 */
export const getMiembrosEquipo = async (): Promise<any[]> => {
  // TODO (Desarrollador B): Implementar obtención de miembros
  // Si está en modo mock, devolver los 24 investigadores desde localStorage (sembrando AUTHORS_DATABASE si está vacío).
  // Si no, consultar la tabla 'miembros_equipo' ordenada por nombre.
  return [];
};

/**
 * Obtiene la lista de papers científicos con sus respectivos autores unidos.
 */
export const getPapers = async (): Promise<any[]> => {
  // TODO (Desarrollador B): Implementar obtención relacional de papers
  // Si está en modo mock, devolver los papers desde localStorage (sembrando PAPERS si está vacío).
  // Si no, consultar 'papers' haciendo join con 'paper_authors' y 'miembros_equipo' en una sola petición.
  // Mapear los coautores anidados al array plano de iniciales (authors: string[]).
  return [];
};

/**
 * Guarda un nuevo paper en Supabase (insertando relaciones M-M) o en LocalStorage en modo mock.
 */
export const savePaper = async (_paperForm: any, _paperFile: File | null, _selectedAuthors: string[]): Promise<void> => {
  // TODO (Desarrollador B): Implementar guardado de paper
  // Si paperFile existe, subirlo a storage ('papers') y obtener la URL pública.
  // Si está en modo mock, añadir a 'gibd_mock_papers' y asociar coautores.
  // Si no, insertar en la tabla 'papers' de Supabase, recuperar el ID asignado,
  // e insertar las filas correspondientes en la tabla intermedia 'paper_authors'.
};

