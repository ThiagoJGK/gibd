-- ==========================================
-- GIBD WEB 2026 - Esquema de Base de Datos Relacional
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
    tag VARCHAR(50) NOT NULL,  -- Ej: 'Publicación', 'Investigación'
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- ==========================================
-- Políticas de Row Level Security (RLS)
-- Para asegurar el acceso público a lecturas y restringir escrituras a administradores
-- ==========================================

ALTER TABLE public.miembros_equipo ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.papers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_authors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.noticias ENABLE ROW LEVEL SECURITY;

-- Políticas de Lectura Abierta (Lectores Anónimos y Públicos)
CREATE POLICY "Permitir lectura publica de miembros" ON public.miembros_equipo 
    FOR SELECT USING (true);

CREATE POLICY "Permitir lectura publica de papers" ON public.papers 
    FOR SELECT USING (true);

CREATE POLICY "Permitir lectura publica de autores_papers" ON public.paper_authors 
    FOR SELECT USING (true);

CREATE POLICY "Permitir lectura publica de noticias" ON public.noticias 
    FOR SELECT USING (true);

-- Políticas de Escritura Protegida (Solo Usuarios Autenticados mediante Supabase Auth)
CREATE POLICY "Permitir escritura solo a autenticados para miembros" ON public.miembros_equipo 
    FOR ALL TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Permitir escritura solo a autenticados para papers" ON public.papers 
    FOR ALL TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Permitir escritura solo a autenticados para autores_papers" ON public.paper_authors 
    FOR ALL TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY "Permitir escritura solo a autenticados para noticias" ON public.noticias 
    FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- ==========================================
-- Semilla de Datos Inicial (Seed Data)
-- Para poblar con miembros básicos en la primera corrida
-- ==========================================

INSERT INTO public.miembros_equipo (initials, name, role, email, linkedin) VALUES
('AP', 'Mg. Andrés Jorge Pascal', 'Docente Investigador', 'pascala@frcu.utn.edu.ar', 'https://linkedin.com/in/andres-pascal-gibd'),
('PC', 'Dra. Patricia R. Cristaldo', 'Docente Investigadora', 'cristaldop@frcu.utn.edu.ar', 'https://linkedin.com/in/patricia-cristaldo'),
('DL', 'Dra. María Daniela López De Luise', 'Docente Investigadora', 'deluisem@frcu.utn.edu.ar', 'https://linkedin.com/in/daniela-lopez-de-luise'),
('TG', 'Thiago Gomez Kehler', 'Investigador', 'thiagogomezkehler@frcu.utn.edu.ar', 'https://linkedin.com/in/thiago-gomez-kehler'),
('LD', 'Luciano Emmanuel Davezac', 'Investigador', 'davezacl@frcu.utn.edu.ar', 'https://linkedin.com/in/luciano-davezac'),
('LC', 'León Castiglioni', 'Investigador', 'castiglionil@frcu.utn.edu.ar', 'https://linkedin.com/in/leon-castiglioni')
ON CONFLICT (initials) DO NOTHING;
