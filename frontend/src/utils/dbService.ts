import { supabase, hasSupabaseConfig } from './supabaseClient';

// Helper para determinar si se debe correr en modo Mock (Desarrollo Local sin conexión)
const checkIfMock = (): boolean => !hasSupabaseConfig;

// ==========================================
// MOCK DATA (SEMILLAS DE RESPALDO LOCAL)
// ==========================================

// Semilla inicial para noticias (Desarrollador A - Emmanuel)
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

// Semilla de autores / investigadores (Desarrollador B - Renato)
const MOCK_AUTHORS = [
  { id: '1', initials: 'AP', name: 'Mg. Andrés Jorge Pascal', role: 'Docente Investigador', email: 'pascala@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/andres-pascal-gibd' },
  { id: '2', initials: 'PC', name: 'Dra. Patricia R. Cristaldo', role: 'Docente Investigadora', email: 'cristaldop@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/patricia-cristaldo' },
  { id: '3', initials: 'DL', name: 'Dra. María Daniela López De Luise', role: 'Docente Investigadora', email: 'deluisem@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/daniela-lopez-de-luise' },
  { id: '4', initials: 'TG', name: 'Thiago Gomez Kehler', role: 'Investigador', email: 'thiagogomezkehler@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/thiago-gomez-kehler' },
  { id: '5', initials: 'MO', name: 'Maximiliano Olivera', role: 'Investigador', email: 'oliveram@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/maximiliano-olivera' },
  { id: '6', initials: 'PS', name: 'Pablo Suarez Lapalma', role: 'Investigador', email: 'suarezp@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/pablo-suarez-lapalma' },
  { id: '7', initials: 'IM', name: 'Iara Martinelli', role: 'Investigadora', email: 'martinellii@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/iara-martinelli' },
  { id: '8', initials: 'MF', name: 'María Emilia Fernandez', role: 'Investigadora', email: 'fernandezm@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/maria-emilia-fernandez' },
  { id: '9', initials: 'LD', name: 'Luciano Emmanuel Davezac', role: 'Investigador', email: 'davezacl@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/luciano-davezac' },
  { id: '10', initials: 'ST', name: 'Sebastián Trossero', role: 'Investigador', email: 'trosseros@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/sebastian-trossero' },
  { id: '11', initials: 'CA', name: 'Claudia M. Álvarez', role: 'Investigadora', email: 'alvarezc@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/claudia-m-alvarez' },
  { id: '12', initials: 'FH', name: 'Fernando Heit', role: 'Investigador', email: 'heitf@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/fernando-heit' },
  { id: '13', initials: 'NP', name: 'Adrián Nicolas Planas', role: 'Docente Investigador', email: 'planasn@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/adrian-nicolas-planas' },
  { id: '14', initials: 'FV', name: 'Florencia Zoe Vidal', role: 'Investigadora', email: 'vidalf@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/florencia-zoe-vidal' },
  { id: '15', initials: 'AB', name: 'Agustina Bonti', role: 'Investigadora', email: 'bontia@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/agustina-bonti' },
  { id: '16', initials: 'LT', name: 'Lucas Francisco Tonelotto', role: 'Investigador', email: 'tonelottol@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/lucas-tonelotto' },
  { id: '17', initials: 'LC', name: 'León Castiglioni', role: 'Investigador', email: 'castiglionil@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/leon-castiglioni' },
  { id: '18', initials: 'FL', name: 'Federico Lederhos', role: 'Investigador', email: 'lederhosf@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/federico-lederhos' },
  { id: '19', initials: 'WC', name: 'Wenceslao Colazo', role: 'Investigador', email: 'colazow@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/wenceslao-colazo' },
  { id: '20', initials: 'SP', name: 'Santiago Poerio Val', role: 'Investigador', email: 'poerios@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/santiago-poerio' },
  { id: '21', initials: 'FS', name: 'Federico Stauber', role: 'Docente Investigador', email: 'stauberf@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/federico-stauber' },
  { id: '22', initials: 'LV', name: 'Luciana G. Valiente', role: 'Investigadora', email: 'valientel@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/luciana-valiente' },
  { id: '23', initials: 'LP', name: 'Lucas La Pietra', role: 'Investigador', email: 'lapietral@frcu.utn.edu.ar', linkedin: 'https://linkedin.com/in/lucas-la-pietra' },
  { id: '24', initials: 'JH', name: 'Dr. Jude Hemanth', role: 'Investigador Externo', email: 'judehemanth@external.utn.edu.ar', linkedin: 'https://linkedin.com/in/jude-hemanth' }
];

// Semilla de papers científicos (Desarrollador B - Renato)
const MOCK_PAPERS_SEED = [
  {
    id: '1',
    title: 'Recuperación de Información de Reglamentación Académica en Español utilizando Modelos del Lenguaje Natural',
    description: 'Investigación sobre el diseño y la sintonización de modelos de lenguaje natural (NLP) aplicados a la consulta y recuperación semántica de reglamentaciones internas en la UTN FRCU.',
    category: 'Procesamiento de Lenguaje Natural',
    date: 'Noviembre 2024 (CoNaIISI 2024)',
    authors: ['AP', 'MO', 'PS', 'IM', 'MF', 'LD', 'TG'],
    image: 'https://images.unsplash.com/photo-1546410531-bb4caa6b424d?auto=format&fit=crop&w=600&q=80',
    url: 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd'
  },
  {
    id: '2',
    title: 'Advanced Variable Tuning and Biases in Chatbot Models: Analysis of the PTAH Prototype',
    description: 'Análisis profundo de la sintonización de parámetros y mitigación de sesgos en modelos de lenguaje conversacionales aplicados al prototipo de agente de IA legal PTAH.',
    category: 'Procesamiento de Lenguaje Natural',
    date: 'Octubre 2024 (IEEE ARGENCON 2024)',
    authors: ['DL', 'AP', 'ST', 'CA', 'FH'],
    image: 'https://images.unsplash.com/photo-1507668077129-56e32842fceb?auto=format&fit=crop&w=600&q=80',
    url: 'https://ieeexplore.ieee.org/document/10705886'
  },
  {
    id: '3',
    title: 'Image Feature Extraction for Similarity Searching Using Transfer Learning with ResNet',
    description: 'Estudio sobre la extracción de descriptores visuales de alta fidelidad mediante transferencia de aprendizaje (ResNet) para optimizar consultas por similitud métrica.',
    category: 'Búsqueda por Similitud',
    date: 'Octubre 2024 (CACIC 2024)',
    authors: ['AP', 'NP', 'FV', 'AB', 'LT', 'LC'],
    image: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=600&q=80',
    url: 'http://sedici.unlp.edu.ar/handle/10915/169192'
  },
  {
    id: '4',
    title: 'Mejorando la Identificación de Marcas de Ganado Vacuno: Redes Siamesas en el Aprendizaje de Funciones de Distancia',
    description: 'Propuesta para optimizar el reconocimiento automático de marcas de propiedad de ganado vacuno en el sector agropecuario mediante aprendizaje de métricas profundas con redes siamesas.',
    category: 'Búsqueda por Similitud',
    date: 'Octubre 2023 (CACIC 2023)',
    authors: ['FS', 'NP', 'AP'],
    image: 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=600&q=80',
    url: 'http://sedici.unlp.edu.ar/handle/10915/161407'
  },
  {
    id: '5',
    title: 'Búsqueda por Similitud de Tatuajes Utilizando Preprocesamiento y Transfer Learning',
    description: 'Desarrollo de un pipeline de preprocesamiento de imágenes combinado con redes convolucionales preentrenadas para la recuperación eficiente de tatuajes forenses.',
    category: 'Búsqueda por Similitud',
    date: 'Noviembre 2024 (CoNaIISI 2024)',
    authors: ['AP', 'NP', 'FL', 'LC', 'WC', 'SP'],
    image: 'https://images.unsplash.com/photo-1546410531-bb4caa6b424d?auto=format&fit=crop&w=600&q=80',
    url: 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd'
  },
  {
    id: '6',
    title: 'Experiencia Ludificada para el desarrollo de Métricas en Gestión de Proyectos',
    description: 'Presentación del modelo y prototipo LudgePI, integrando dinámicas lúdicas de gamificación para facilitar la recopilación y análisis transversal de métricas de calidad en proyectos.',
    category: 'Gestión de Proyectos',
    date: 'Noviembre 2023 (CoNaIISI 2023)',
    authors: ['PC', 'DL', 'LT', 'LV'],
    image: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=600&q=80',
    url: 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd'
  },
  {
    id: '7',
    title: 'Metrics for the Systematic Evaluation of Software Project Management Methodologies',
    description: 'Definición de un marco de métricas transversales basado en minería de datos para evaluar sistemáticamente la alineación y desempeño de metodologías híbridas y ágiles.',
    category: 'Gestión de Proyectos',
    date: '2021 (Global Research and Development Journal)',
    authors: ['PC', 'DL', 'LP', 'AB', 'JH'],
    image: 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=600&q=80',
    url: 'https://www.grdjournals.com/article?paper=GRDJEV06I050009'
  }
];

// ==========================================
// SUBIDA DE ARCHIVOS A STORAGE
// ==========================================

export const uploadFile = async (file: File, bucket: 'noticias' | 'papers'): Promise<string> => {
  const isMock = checkIfMock();
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

// Alias para compatibilidad con el código de Emmanuel
export const uploadToStorage = uploadFile;

// ==========================================
// MÉTODOS PARA DESARROLLADOR A (MÓDULO NOTICIAS)
// ==========================================

export const getNoticias = async (): Promise<any[]> => {
  const isMock = checkIfMock();
  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_noticias');
    if (localData) return JSON.parse(localData);
    
    // Sembrar la semilla inicial
    localStorage.setItem('gibd_mock_noticias', JSON.stringify(MOCK_NOTICIAS_SEED));
    return MOCK_NOTICIAS_SEED;
  }

  const { data, error } = await supabase
    .from('noticias')
    .select('*')
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data || [];
};

export const saveNoticia = async (noticiaForm: any, noticiaFile: File | null): Promise<void> => {
  const isMock = checkIfMock();
  let imageUrl = noticiaForm.imageUrl;

  if (noticiaFile) {
    imageUrl = await uploadFile(noticiaFile, 'noticias');
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

export const getMiembrosEquipo = async (): Promise<any[]> => {
  const isMock = checkIfMock();
  
  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_miembros_equipo');
    if (localData) return JSON.parse(localData);
    
    localStorage.setItem('gibd_mock_miembros_equipo', JSON.stringify(MOCK_AUTHORS));
    return MOCK_AUTHORS;
  }
  
  const { data, error } = await supabase
    .from('miembros_equipo')
    .select('id, initials, name, role, email, linkedin')
    .order('name');

  if (error) throw error;
  return data || [];
};

export const getPapers = async (): Promise<any[]> => {
  const isMock = checkIfMock();

  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_papers');
    if (localData) return JSON.parse(localData);
    
    localStorage.setItem('gibd_mock_papers', JSON.stringify(MOCK_PAPERS_SEED));
    return MOCK_PAPERS_SEED;
  }

  const { data, error } = await supabase
    .from('papers')
    .select(`
      id, title, description, category, date, url, image_url,
      paper_authors (
        miembros_equipo ( initials )
      )
    `)
    .order('created_at', { ascending: false });
  
  if (error) throw error;

  return (data || []).map((paper: any) => ({
    id: paper.id,
    title: paper.title,
    description: paper.description,
    category: paper.category,
    date: paper.date,
    url: paper.url,
    image: paper.image_url || 'https://images.unsplash.com/photo-1546410531-bb4caa6b424d?auto=format&fit=crop&w=600&q=80',
    authors: (paper.paper_authors || [])
      .map((pa: any) => pa.miembros_equipo?.initials)
      .filter(Boolean)
  }));
};

export const savePaper = async (paperForm: any, paperFile: File | null, selectedAuthors: string[]): Promise<void> => {
  const isMock = checkIfMock();
  let finalPdfUrl = paperForm.url;

  if (paperFile) {
    finalPdfUrl = await uploadFile(paperFile, 'papers');
  }

  const formattedDate = paperForm.date || new Date().toLocaleDateString('es-ES', { month: 'long', year: 'numeric' });

  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_papers');
    const papers = localData ? JSON.parse(localData) : [];
    
    // Obtener los coautores por sus IDs y mapear a iniciales
    const authorsList = await getMiembrosEquipo();
    const mappedInitials = selectedAuthors
      .map(id => authorsList.find((a: any) => a.id === id)?.initials)
      .filter(Boolean);

    papers.unshift({
      id: Date.now().toString(),
      title: paperForm.title,
      description: paperForm.description,
      category: paperForm.category,
      date: formattedDate,
      url: finalPdfUrl,
      image: paperForm.imageUrl && !paperForm.imageUrl.startsWith('AI') 
        ? paperForm.imageUrl 
        : 'https://images.unsplash.com/photo-1546410531-bb4caa6b424d?auto=format&fit=crop&w=600&q=80',
      authors: mappedInitials
    });

    localStorage.setItem('gibd_mock_papers', JSON.stringify(papers));
    return;
  }

  // 1. Insertar el paper científico
  const { data: insertedPaper, error: paperError } = await supabase
    .from('papers')
    .insert({
      title: paperForm.title,
      description: paperForm.description,
      category: paperForm.category,
      date: formattedDate,
      url: finalPdfUrl,
      image_url: paperForm.imageUrl && !paperForm.imageUrl.startsWith('AI') ? paperForm.imageUrl : null
    })
    .select()
    .single();

  if (paperError) throw paperError;

  // 2. Insertar relaciones Many-to-Many en paper_authors
  if (selectedAuthors.length > 0 && insertedPaper) {
    const relationInserts = selectedAuthors.map(authorId => ({
      paper_id: insertedPaper.id,
      author_id: authorId
    }));

    const { error: authorRelError } = await supabase
      .from('paper_authors')
      .insert(relationInserts);

    if (authorRelError) throw authorRelError;
  }
};

export const updatePaper = async (paperId: string | number, paperForm: any, paperFile: File | null, selectedAuthors: string[]): Promise<void> => {
  const isMock = checkIfMock();
  let finalPdfUrl = paperForm.url;

  if (paperFile) {
    finalPdfUrl = await uploadFile(paperFile, 'papers');
  }

  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_papers');
    if (!localData) return;
    
    const papers = JSON.parse(localData);
    const index = papers.findIndex((p: any) => p.id.toString() === paperId.toString());
    
    if (index !== -1) {
      const authorsList = await getMiembrosEquipo();
      const mappedInitials = selectedAuthors
        .map(id => authorsList.find((a: any) => a.id === id)?.initials)
        .filter(Boolean);

      papers[index] = {
        ...papers[index],
        title: paperForm.title,
        description: paperForm.description,
        category: paperForm.category,
        date: paperForm.date,
        url: finalPdfUrl,
        image: paperForm.imageUrl && !paperForm.imageUrl.startsWith('AI') 
          ? paperForm.imageUrl 
          : papers[index].image,
        authors: mappedInitials
      };
      
      localStorage.setItem('gibd_mock_papers', JSON.stringify(papers));
    }
    return;
  }

  // 1. Actualizar el registro del paper
  const { error: paperError } = await supabase
    .from('papers')
    .update({
      title: paperForm.title,
      description: paperForm.description,
      category: paperForm.category,
      date: paperForm.date,
      url: finalPdfUrl,
      image_url: paperForm.imageUrl && !paperForm.imageUrl.startsWith('AI') ? paperForm.imageUrl : null
    })
    .eq('id', paperId);

  if (paperError) throw paperError;

  // 2. Eliminar relaciones co-autores antiguas
  const { error: deleteRelError } = await supabase
    .from('paper_authors')
    .delete()
    .eq('paper_id', paperId);

  if (deleteRelError) throw deleteRelError;

  // 3. Insertar relaciones co-autores nuevas
  if (selectedAuthors.length > 0) {
    const relationInserts = selectedAuthors.map(authorId => ({
      paper_id: paperId,
      author_id: authorId
    }));

    const { error: authorRelError } = await supabase
      .from('paper_authors')
      .insert(relationInserts);

    if (authorRelError) throw authorRelError;
  }
};

export const deletePaper = async (paperId: string | number): Promise<void> => {
  const isMock = checkIfMock();

  if (isMock) {
    const localData = localStorage.getItem('gibd_mock_papers');
    if (!localData) return;
    
    const papers = JSON.parse(localData);
    const filteredPapers = papers.filter((p: any) => p.id.toString() !== paperId.toString());
    
    localStorage.setItem('gibd_mock_papers', JSON.stringify(filteredPapers));
    return;
  }

  // Borrado directo en Supabase. ON DELETE CASCADE se encarga de la tabla intermedia.
  const { error } = await supabase
    .from('papers')
    .delete()
    .eq('id', paperId);

  if (error) throw error;
};
