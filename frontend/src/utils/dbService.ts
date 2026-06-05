// import { supabase } from './supabaseClient';

// ==========================================
// MÉTODOS PARA DESARROLLADOR A (MÓDULO NOTICIAS)
// ==========================================

/**
 * Obtiene las noticias de la base de datos o LocalStorage si está en modo mock.
 */
export const getNoticias = async (): Promise<any[]> => {
  // TODO (Desarrollador A): Implementar lógica de obtención dinámica
  // Si no hay variables de Supabase configuradas, leer de localStorage (sembrando NEWS_ITEMS si está vacío).
  // Si hay conexión a Supabase, consultar la tabla 'noticias' ordenada por created_at DESC.
  return [];
};

/**
 * Guarda una nueva noticia en Supabase o en LocalStorage si está en modo mock.
 */
export const saveNoticia = async (_noticiaForm: any, _noticiaFile: File | null): Promise<void> => {
  // TODO (Desarrollador A): Implementar lógica de guardado
  // Si noticiaFile existe, subirlo a storage ('noticias') y obtener la URL pública.
  // Si está en modo mock, añadir al array 'gibd_mock_noticias' en localStorage.
  // Si no, insertar en la tabla 'noticias' de Supabase.
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

