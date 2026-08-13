import { createClient } from '@supabase/supabase-js';

// Cargar variables de entorno configuradas en .env.local (o inyectadas por Vercel/Hosting)
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';
const hasSupabaseConfig = Boolean(
  supabaseUrl &&
  supabaseAnonKey &&
  supabaseUrl !== 'YOUR_SUPABASE_URL'
);

const mockSupabaseClient = {
  auth: {
    getSession: async () => ({ data: { session: null }, error: null }),
    onAuthStateChange: () => ({
      data: { subscription: { unsubscribe: () => undefined } }
    }),
    signInWithPassword: async () => ({ data: null, error: new Error('Supabase no configurado') }),
    signInWithOAuth: async () => ({ data: null, error: new Error('Supabase no configurado') }),
    signOut: async () => ({ error: null })
  },
  storage: {
    from: () => ({
      upload: async () => ({ error: new Error('Supabase no configurado') }),
      getPublicUrl: () => ({ data: { publicUrl: '' } })
    })
  },
  from: () => ({
    select: async () => ({ data: [], error: null }),
    insert: async () => ({ data: null, error: null }),
    update: async () => ({ data: null, error: null }),
    delete: async () => ({ data: null, error: null })
  })
} as any;

// Inicializar y exportar el cliente único de Supabase.
// Si faltan las variables de entorno, se activa el modo mock para que la UI siga funcionando.
export const supabase = hasSupabaseConfig ? createClient(supabaseUrl, supabaseAnonKey) : mockSupabaseClient;

export default supabase;
