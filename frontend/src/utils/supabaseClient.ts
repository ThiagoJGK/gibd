import { createClient } from '@supabase/supabase-js';

// Cargar variables de entorno configuradas en .env.local (o inyectadas por Vercel/Hosting)
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

// Inicializar y exportar el cliente único de Supabase
export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export default supabase;
