import { createClient } from '@supabase/supabase-js';

// Cargar variables de entorno configuradas en .env.local (o inyectadas por Vercel/Hosting)
// Si no están configuradas, usamos credenciales dummy válidas para evitar que createClient falle catastróficamente.
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://placeholder-project.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'placeholder-anon-key';

// Inicializar y exportar el cliente único de Supabase
export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export default supabase;

