-- ==========================================
-- GIBD WEB 2026 - Esquema de Base de Datos Relacional Completo
-- Motor: PostgreSQL (Supabase)
-- ==========================================

-- 1. Tabla de Miembros del Equipo / Autores
CREATE TABLE IF NOT EXISTS public.miembros_equipo (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    initials VARCHAR(10) UNIQUE NOT NULL, -- Ej: 'AP', 'TG', 'DL'
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    email TEXT UNIQUE,
    linkedin TEXT,
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Tabla de Papers Científicos
CREATE TABLE IF NOT EXISTS public.papers (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    date VARCHAR(100) NOT NULL, -- Ej: 'Noviembre 2024 (CoNaIISI 2024)'
    image_url TEXT,
    url TEXT NOT NULL, -- Enlace de acceso / descarga del PDF
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Tabla Intermedia (Muchos a Muchos) de Autores por Paper
CREATE TABLE IF NOT EXISTS public.paper_authors (
    paper_id UUID REFERENCES public.papers(id) ON DELETE CASCADE,
    author_id UUID REFERENCES public.miembros_equipo(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_id, author_id)
);

-- 4. Tabla de Noticias / Novedades de la Landing Page
CREATE TABLE IF NOT EXISTS public.noticias (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    date VARCHAR(50) NOT NULL, -- Ej: 'Nov 2024'
    tag VARCHAR(50) NOT NULL,  -- Ej: 'Publicación', 'Investigación', 'Conferencia'
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- ==========================================
-- Políticas de Row Level Security (RLS)
-- Acceso público a lecturas y escrituras autenticadas
-- ==========================================

ALTER TABLE public.miembros_equipo ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.papers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_authors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.noticias ENABLE ROW LEVEL SECURITY;

-- Políticas de Lectura Abierta (Público)
CREATE POLICY "Permitir lectura publica de miembros" ON public.miembros_equipo FOR SELECT USING (true);
CREATE POLICY "Permitir lectura publica de papers" ON public.papers FOR SELECT USING (true);
CREATE POLICY "Permitir lectura publica de autores_papers" ON public.paper_authors FOR SELECT USING (true);
CREATE POLICY "Permitir lectura publica de noticias" ON public.noticias FOR SELECT USING (true);

-- Políticas de Escritura Protegida (Autenticados / CMS Admin)
CREATE POLICY "Permitir escritura solo a autenticados para miembros" ON public.miembros_equipo FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Permitir escritura solo a autenticados para papers" ON public.papers FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Permitir escritura solo a autenticados para autores_papers" ON public.paper_authors FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Permitir escritura solo a autenticados para noticias" ON public.noticias FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- ==========================================
-- Storage Buckets (Para PDFs e Imágenes)
-- ==========================================

INSERT INTO storage.buckets (id, name, public) 
VALUES ('noticias', 'noticias', true), ('papers', 'papers', true)
ON CONFLICT (id) DO NOTHING;

-- Políticas de Storage
CREATE POLICY "Acceso publico de lectura en storage noticias" ON storage.objects FOR SELECT USING (bucket_id = 'noticias');
CREATE POLICY "Acceso publico de lectura en storage papers" ON storage.objects FOR SELECT USING (bucket_id = 'papers');
CREATE POLICY "Escritura autenticada en storage noticias" ON storage.objects FOR INSERT TO authenticated WITH CHECK (bucket_id = 'noticias');
CREATE POLICY "Escritura autenticada en storage papers" ON storage.objects FOR INSERT TO authenticated WITH CHECK (bucket_id = 'papers');

-- ==========================================
-- Semilla de Datos Inicial (Seed Data)
-- ==========================================

-- 1. Los 24 Autores / Investigadores del GIBD
INSERT INTO public.miembros_equipo (initials, name, role, email, linkedin) VALUES
('AP', 'Mg. Andrés Jorge Pascal', 'Docente Investigador', 'pascala@frcu.utn.edu.ar', 'https://linkedin.com/in/andres-pascal-gibd'),
('PC', 'Dra. Patricia R. Cristaldo', 'Docente Investigadora', 'cristaldop@frcu.utn.edu.ar', 'https://linkedin.com/in/patricia-cristaldo'),
('DL', 'Dra. María Daniela López De Luise', 'Docente Investigadora', 'deluisem@frcu.utn.edu.ar', 'https://linkedin.com/in/daniela-lopez-de-luise'),
('TG', 'Thiago Gomez Kehler', 'Investigador', 'thiagogomezkehler@frcu.utn.edu.ar', 'https://linkedin.com/in/thiago-gomez-kehler'),
('MO', 'Maximiliano Olivera', 'Investigador', 'oliveram@frcu.utn.edu.ar', 'https://linkedin.com/in/maximiliano-olivera'),
('PS', 'Pablo Suarez Lapalma', 'Investigador', 'suarezp@frcu.utn.edu.ar', 'https://linkedin.com/in/pablo-suarez-lapalma'),
('IM', 'Iara Martinelli', 'Investigadora', 'martinellii@frcu.utn.edu.ar', 'https://linkedin.com/in/iara-martinelli'),
('MF', 'María Emilia Fernandez', 'Investigadora', 'fernandezm@frcu.utn.edu.ar', 'https://linkedin.com/in/maria-emilia-fernandez'),
('LD', 'Luciano Emmanuel Davezac', 'Investigador', 'davezacl@frcu.utn.edu.ar', 'https://linkedin.com/in/luciano-davezac'),
('ST', 'Sebastián Trossero', 'Investigador', 'trosseros@frcu.utn.edu.ar', 'https://linkedin.com/in/sebastian-trossero'),
('CA', 'Claudia M. Álvarez', 'Investigadora', 'alvarezc@frcu.utn.edu.ar', 'https://linkedin.com/in/claudia-m-alvarez'),
('FH', 'Fernando Heit', 'Investigador', 'heitf@frcu.utn.edu.ar', 'https://linkedin.com/in/fernando-heit'),
('NP', 'Adrián Nicolas Planas', 'Docente Investigador', 'planasn@frcu.utn.edu.ar', 'https://linkedin.com/in/adrian-nicolas-planas'),
('FV', 'Florencia Zoe Vidal', 'Investigadora', 'vidalf@frcu.utn.edu.ar', 'https://linkedin.com/in/florencia-zoe-vidal'),
('AB', 'Agustina Bonti', 'Investigadora', 'bontia@frcu.utn.edu.ar', 'https://linkedin.com/in/agustina-bonti'),
('LT', 'Lucas Francisco Tonelotto', 'Investigador', 'tonelottol@frcu.utn.edu.ar', 'https://linkedin.com/in/lucas-tonelotto'),
('LC', 'León Castiglioni', 'Investigador', 'castiglionil@frcu.utn.edu.ar', 'https://linkedin.com/in/leon-castiglioni'),
('FL', 'Federico Lederhos', 'Investigador', 'lederhosf@frcu.utn.edu.ar', 'https://linkedin.com/in/federico-lederhos'),
('WC', 'Wenceslao Colazo', 'Investigador', 'colazow@frcu.utn.edu.ar', 'https://linkedin.com/in/wenceslao-colazo'),
('SP', 'Santiago Poerio Val', 'Investigador', 'poerios@frcu.utn.edu.ar', 'https://linkedin.com/in/santiago-poerio'),
('FS', 'Federico Stauber', 'Docente Investigador', 'stauberf@frcu.utn.edu.ar', 'https://linkedin.com/in/federico-stauber'),
('LV', 'Luciana G. Valiente', 'Investigadora', 'valientel@frcu.utn.edu.ar', 'https://linkedin.com/in/luciana-valiente'),
('LP', 'Lucas La Pietra', 'Investigador', 'lapietral@frcu.utn.edu.ar', 'https://linkedin.com/in/lucas-la-pietra'),
('JH', 'Dr. Jude Hemanth', 'Investigador Externo', 'judehemanth@external.utn.edu.ar', 'https://linkedin.com/in/jude-hemanth')
ON CONFLICT (initials) DO NOTHING;

-- 2. Noticias Semilla
INSERT INTO public.noticias (title, content, tag, date) VALUES
('Presentación en CoNaIISI 2024 de la investigación en Recuperación de Información de Reglamentación Académica en Español utilizando NLP.', 'Se presentó el trabajo de investigación desarrollado por el equipo sobre Recuperación de Información de Reglamentación Académica en Español utilizando técnicas avanzadas de Procesamiento del Lenguaje Natural.', 'Publicación', 'Nov 2024'),
('Aceptación y publicación en ARGENCON 2024 del paper "Advanced Variable Tuning and Biases in Chatbot Models: Analysis of the PTAH Prototype".', 'El paper del prototipo conversacional PTAH fue publicado en ARGENCON 2024. Este artículo presenta un análisis exhaustivo sobre el sesgo y ajuste de variables en modelos de chatbot.', 'Investigación', 'Oct 2024'),
('Exposición en CACIC 2024: "Image Feature Extraction for Similarity Searching Using Transfer Learning with ResNet".', 'El equipo expuso el trabajo que detalla el uso de Transfer Learning con ResNet para la extracción de características visuales en búsquedas por similitud.', 'Conferencia', 'Oct 2024');

-- 3. Papers Semilla
INSERT INTO public.papers (id, title, description, category, date, url, image_url) VALUES
('11111111-1111-1111-1111-111111111111', 'Recuperación de Información de Reglamentación Académica en Español utilizando Modelos del Lenguaje Natural', 'Investigación sobre el diseño y la sintonización de modelos de lenguaje natural (NLP) aplicados a la consulta y recuperación semántica de reglamentaciones internas en la UTN FRCU.', 'Procesamiento de Lenguaje Natural', 'Noviembre 2024 (CoNaIISI 2024)', 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd', 'https://images.unsplash.com/photo-1546410531-bb4caa6b424d?auto=format&fit=crop&w=600&q=80'),
('22222222-2222-2222-2222-222222222222', 'Advanced Variable Tuning and Biases in Chatbot Models: Analysis of the PTAH Prototype', 'Análisis profundo de la sintonización de parámetros y mitigación de sesgos en modelos de lenguaje conversacionales aplicados al prototipo de agente de IA legal PTAH.', 'Procesamiento de Lenguaje Natural', 'Octubre 2024 (IEEE ARGENCON 2024)', 'https://ieeexplore.ieee.org/document/10705886', 'https://images.unsplash.com/photo-1507668077129-56e32842fceb?auto=format&fit=crop&w=600&q=80'),
('33333333-3333-3333-3333-333333333333', 'Image Feature Extraction for Similarity Searching Using Transfer Learning with ResNet', 'Estudio sobre la extracción de descriptores visuales de alta fidelidad mediante transferencia de aprendizaje (ResNet) para optimizar consultas por similitud métrica.', 'Búsqueda por Similitud', 'Octubre 2024 (CACIC 2024)', 'http://sedici.unlp.edu.ar/handle/10915/169192', 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=600&q=80'),
('44444444-4444-4444-4444-444444444444', 'Mejorando la Identificación de Marcas de Ganado Vacuno: Redes Siamesas en el Aprendizaje de Funciones de Distancia', 'Propuesta para optimizar el reconocimiento automático de marcas de propiedad de ganado vacuno en el sector agropecuario mediante aprendizaje de métricas profundas con redes siamesas.', 'Búsqueda por Similitud', 'Octubre 2023 (CACIC 2023)', 'http://sedici.unlp.edu.ar/handle/10915/161407', 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=600&q=80'),
('55555555-5555-5555-5555-555555555555', 'Búsqueda por Similitud de Tatuajes Utilizando Preprocesamiento y Transfer Learning', 'Desarrollo de un pipeline de preprocesamiento de imágenes combinado con redes convolucionales preentrenadas para la recuperación eficiente de tatuajes forenses.', 'Búsqueda por Similitud', 'Noviembre 2024 (CoNaIISI 2024)', 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd', 'https://images.unsplash.com/photo-1546410531-bb4caa6b424d?auto=format&fit=crop&w=600&q=80'),
('66666666-6666-6666-6666-666666666666', 'Experiencia Ludificada para el desarrollo de Métricas en Gestión de Proyectos', 'Presentación del modelo y prototipo LudgePI, integrando dinámicas lúdicas de gamificación para facilitar la recopilación y análisis transversal de métricas de calidad en proyectos.', 'Gestión de Proyectos', 'Noviembre 2023 (CoNaIISI 2023)', 'https://frcu.utn.edu.ar/index.php/investigacion-gibd/publicaciones-gibd', 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=600&q=80'),
('77777777-7777-7777-7777-777777777777', 'Metrics for the Systematic Evaluation of Software Project Management Methodologies', 'Definición de un marco de métricas transversales basado en minería de datos para evaluar sistemáticamente la alineación y desempeño de metodologías híbridas y ágiles.', 'Gestión de Proyectos', '2021 (Global Research and Development Journal)', 'https://www.grdjournals.com/article?paper=GRDJEV06I050009', 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=600&q=80')
ON CONFLICT (id) DO NOTHING;

-- 4. Relaciones de Co-autores por Paper
-- Paper 1: AP, MO, PS, IM, MF, LD, TG
INSERT INTO public.paper_authors (paper_id, author_id)
SELECT '11111111-1111-1111-1111-111111111111', id FROM public.miembros_equipo WHERE initials IN ('AP', 'MO', 'PS', 'IM', 'MF', 'LD', 'TG')
ON CONFLICT DO NOTHING;

-- Paper 2: DL, AP, ST, CA, FH
INSERT INTO public.paper_authors (paper_id, author_id)
SELECT '22222222-2222-2222-2222-222222222222', id FROM public.miembros_equipo WHERE initials IN ('DL', 'AP', 'ST', 'CA', 'FH')
ON CONFLICT DO NOTHING;

-- Paper 3: AP, NP, FV, AB, LT, LC
INSERT INTO public.paper_authors (paper_id, author_id)
SELECT '33333333-3333-3333-3333-333333333333', id FROM public.miembros_equipo WHERE initials IN ('AP', 'NP', 'FV', 'AB', 'LT', 'LC')
ON CONFLICT DO NOTHING;

-- Paper 4: FS, NP, AP
INSERT INTO public.paper_authors (paper_id, author_id)
SELECT '44444444-4444-4444-4444-444444444444', id FROM public.miembros_equipo WHERE initials IN ('FS', 'NP', 'AP')
ON CONFLICT DO NOTHING;

-- Paper 5: AP, NP, FL, LC, WC, SP
INSERT INTO public.paper_authors (paper_id, author_id)
SELECT '55555555-5555-5555-5555-555555555555', id FROM public.miembros_equipo WHERE initials IN ('AP', 'NP', 'FL', 'LC', 'WC', 'SP')
ON CONFLICT DO NOTHING;

-- Paper 6: PC, DL, LT, LV
INSERT INTO public.paper_authors (paper_id, author_id)
SELECT '66666666-6666-6666-6666-666666666666', id FROM public.miembros_equipo WHERE initials IN ('PC', 'DL', 'LT', 'LV')
ON CONFLICT DO NOTHING;

-- Paper 7: PC, DL, LP, AB, JH
INSERT INTO public.paper_authors (paper_id, author_id)
SELECT '77777777-7777-7777-7777-777777777777', id FROM public.miembros_equipo WHERE initials IN ('PC', 'DL', 'LP', 'AB', 'JH')
ON CONFLICT DO NOTHING;
