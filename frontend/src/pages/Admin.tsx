import React, { useState, useEffect } from 'react';
import { 
  Lock, Mail, Key, Sparkles, Upload, Database, Activity, 
  CheckCircle2, XCircle, AlertCircle, LogOut, Globe, 
  BookOpen, Cpu, RefreshCw, FileUp, 
  Check, UserCheck, AlertTriangle
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { supabase } from '../utils/supabaseClient';
import { playWaterDrip, playRubberSnap } from '../utils/audio';
import './Admin.css';

// Interfaces de datos
interface Author {
  id: string;
  initials: string;
  name: string;
  role: string;
}

export const Admin: React.FC = () => {
  // Estado de Autenticación
  const [session, setSession] = useState<any>(null);
  const [authLoading, setAuthLoading] = useState<boolean>(true);
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [isMockMode, setIsMockMode] = useState<boolean>(false);

  // Estados del CMS y Pestañas
  const [activeTab, setActiveTab] = useState<'noticias' | 'papers' | 'monitoreo'>('noticias');
  
  // Listas de la base de datos
  const [authors, setAuthors] = useState<Author[]>([]);
  const [loadingAuthors, setLoadingAuthors] = useState<boolean>(false);

  // Estados de formularios
  const [saving, setSaving] = useState<boolean>(false);

  // 1. Formulario de Noticias
  const [noticiaForm, setNoticiaForm] = useState({
    title: '',
    content: '',
    tag: 'Novedad',
    date: '',
    imageUrl: ''
  });
  const [noticiaFile, setNoticiaFile] = useState<File | null>(null);
  const [noticiaDragActive, setNoticiaDragActive] = useState<boolean>(false);

  // 2. Formulario de Papers
  const [paperForm, setPaperForm] = useState({
    title: '',
    description: '',
    category: 'Inteligencia Artificial',
    date: '',
    url: '',
    imageUrl: '',
    paperText: '' // Campo auxiliar para análisis con Gemini
  });
  const [selectedAuthors, setSelectedAuthors] = useState<string[]>([]);
  const [paperFile, setPaperFile] = useState<File | null>(null);
  const [paperDragActive, setPaperDragActive] = useState<boolean>(false);
  const [analyzingPaper, setAnalyzingPaper] = useState<boolean>(false);

  // 3. Estado de Monitoreo
  const [backendHealth, setBackendHealth] = useState<{
    status: string;
    service: string;
    gemini_active: boolean;
    loading: boolean;
    error: boolean;
  }>({
    status: 'Desconocido',
    service: 'GIBD API Gateway',
    gemini_active: false,
    loading: true,
    error: false
  });

  // Efecto para inicializar sesión y monitorear cambios
  useEffect(() => {
    // Comprobar si las variables de entorno de Supabase están configuradas
    const hasSupabaseKeys = 
      import.meta.env.VITE_SUPABASE_URL && 
      import.meta.env.VITE_SUPABASE_ANON_KEY &&
      import.meta.env.VITE_SUPABASE_URL !== 'YOUR_SUPABASE_URL';

    if (!hasSupabaseKeys) {
      console.warn("Supabase keys are missing. Running in developer mock-auth mode.");
      setIsMockMode(true);
      // Cargar mock session si existe en localStorage
      const mockSession = localStorage.getItem('gibd_mock_session');
      if (mockSession) {
        setSession(JSON.parse(mockSession));
      }
      setAuthLoading(false);
    } else {
      // Live Supabase Mode
      supabase.auth.getSession().then(({ data: { session } }) => {
        setSession(session);
        setAuthLoading(false);
      }).catch((err) => {
        console.error("Error fetching session, falling back to mock mode:", err);
        setIsMockMode(true);
        setAuthLoading(false);
      });

      const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
        setSession(session);
      });

      return () => {
        subscription.unsubscribe();
      };
    }
  }, []);

  // Cargar autores de Supabase / Mock al autenticarse
  useEffect(() => {
    if (session) {
      fetchAuthors();
      checkBackendHealth();
    }
  }, [session, isMockMode]);

  const fetchAuthors = async () => {
    setLoadingAuthors(true);
    if (isMockMode) {
      // Mock Autores
      setTimeout(() => {
        setAuthors([
          { id: '1', initials: 'AP', name: 'Mg. Andrés Jorge Pascal', role: 'Docente Investigador' },
          { id: '2', initials: 'PC', name: 'Dra. Patricia R. Cristaldo', role: 'Docente Investigadora' },
          { id: '3', initials: 'DL', name: 'Dra. María Daniela López De Luise', role: 'Docente Investigadora' },
          { id: '4', initials: 'TG', name: 'Thiago Gomez Kehler', role: 'Investigador' },
          { id: '5', initials: 'LD', name: 'Luciano Emmanuel Davezac', role: 'Investigador' },
          { id: '6', initials: 'LC', name: 'León Castiglioni', role: 'Investigador' }
        ]);
        setLoadingAuthors(false);
      }, 400);
    } else {
      try {
        const { data, error } = await supabase
          .from('miembros_equipo')
          .select('id, initials, name, role')
          .order('name');
        
        if (error) throw error;
        setAuthors(data || []);
      } catch (err: any) {
        console.error("Error fetching authors:", err.message);
        // Fallback a mock si falla la tabla
        setAuthors([
          { id: '1', initials: 'AP', name: 'Mg. Andrés Jorge Pascal', role: 'Docente Investigador' },
          { id: '2', initials: 'TG', name: 'Thiago Gomez Kehler', role: 'Investigador' }
        ]);
      } finally {
        setLoadingAuthors(false);
      }
    }
  };

  const checkBackendHealth = async () => {
    setBackendHealth(prev => ({ ...prev, loading: true, error: false }));
    try {
      const res = await fetch('http://localhost:8000/api/v1/health');
      if (!res.ok) throw new Error('API server error');
      const data = await res.json();
      setBackendHealth({
        status: data.status,
        service: data.service,
        gemini_active: data.gemini_active,
        loading: false,
        error: false
      });
    } catch (err) {
      console.warn("Backend not running or unreachable:", err);
      setBackendHealth({
        status: 'Inaccesible',
        service: 'GIBD API Gateway',
        gemini_active: false,
        loading: false,
        error: true
      });
    }
  };

  // Manejadores de Autenticación
  const handleEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setSuccessMsg(null);
    playWaterDrip();

    if (!email || !password) {
      setErrorMsg('Por favor completa todos los campos.');
      return;
    }

    if (isMockMode) {
      // Mock Login
      if (email === 'admin@gibd.utn.edu.ar' && password === 'admin123') {
        const fakeSession = {
          user: {
            email: 'admin@gibd.utn.edu.ar',
            role: 'authenticated',
            user_metadata: { full_name: 'GIBD Administrador Local' }
          }
        };
        localStorage.setItem('gibd_mock_session', JSON.stringify(fakeSession));
        setSession(fakeSession);
        setSuccessMsg('Inicio de sesión exitoso (Modo Desarrollador Local).');
      } else {
        setErrorMsg('Credenciales inválidas en modo local. Usa: admin@gibd.utn.edu.ar / admin123');
      }
    } else {
      // Real Supabase Login
      try {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password
        });
        if (error) throw error;
        setSuccessMsg('Inicio de sesión exitoso.');
      } catch (err: any) {
        setErrorMsg(err.message || 'Error al iniciar sesión.');
      }
    }
  };

  const handleGoogleLogin = async () => {
    playWaterDrip();
    setErrorMsg(null);

    if (isMockMode) {
      // Mock Google Login
      const fakeSession = {
        user: {
          email: 'thiago.gk.admin@gmail.com',
          role: 'authenticated',
          user_metadata: { full_name: 'Thiago Gomez Kehler (Google Mock)' }
        }
      };
      localStorage.setItem('gibd_mock_session', JSON.stringify(fakeSession));
      setSession(fakeSession);
      setSuccessMsg('Inicio de sesión con Google exitoso (Mock).');
    } else {
      try {
        const { error } = await supabase.auth.signInWithOAuth({
          provider: 'google',
          options: {
            redirectTo: window.location.origin + '/admin'
          }
        });
        if (error) throw error;
      } catch (err: any) {
        setErrorMsg(err.message || 'Error al conectar con Google.');
      }
    }
  };

  const handleLogout = async () => {
    playWaterDrip();
    if (isMockMode) {
      localStorage.removeItem('gibd_mock_session');
      setSession(null);
    } else {
      await supabase.auth.signOut();
      setSession(null);
    }
    setSuccessMsg('Sesión cerrada con éxito.');
  };

  // Drag and Drop y subida de archivos
  const handleDrag = (e: React.DragEvent, type: 'noticia' | 'paper') => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      if (type === 'noticia') setNoticiaDragActive(true);
      else setPaperDragActive(true);
    } else if (e.type === "dragleave") {
      if (type === 'noticia') setNoticiaDragActive(false);
      else setPaperDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent, type: 'noticia' | 'paper') => {
    e.preventDefault();
    e.stopPropagation();
    if (type === 'noticia') {
      setNoticiaDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        setNoticiaFile(e.dataTransfer.files[0]);
        playRubberSnap();
      }
    } else {
      setPaperDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        setPaperFile(e.dataTransfer.files[0]);
        playRubberSnap();
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'noticia' | 'paper') => {
    if (e.target.files && e.target.files[0]) {
      if (type === 'noticia') setNoticiaFile(e.target.files[0]);
      else setPaperFile(e.target.files[0]);
      playRubberSnap();
    }
  };

  // Subir archivo real a Supabase Storage
  const uploadToStorage = async (file: File, bucket: 'noticias' | 'papers'): Promise<string> => {
    if (isMockMode) {
      // Simular subida en modo local, retornando una URL mockup de Vercel/Supabase
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

  // Invocación a Gemini AI vía el backend FastAPI local
  const optimizeWithGemini = async () => {
    if (!paperForm.paperText) {
      alert('Por favor introduce el contenido textual del Paper a analizar.');
      return;
    }

    setAnalyzingPaper(true);
    playWaterDrip();
    
    try {
      const response = await fetch('http://localhost:8000/api/v1/ai/analyze-paper', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ paper_text: paperForm.paperText })
      });

      if (!response.ok) {
        throw new Error('El backend de IA no respondió exitosamente.');
      }

      const result = await response.json();
      
      setPaperForm(prev => ({
        ...prev,
        description: result.abstract,
        // Poner en el log de prompt la sugerencia de imagen
        imageUrl: `AI Suggested prompt: ${result.image_prompt}`
      }));
      
      playRubberSnap();
      alert('¡Optimización de IA completada! Se autocompletó la descripción académica.');
    } catch (err: any) {
      console.error(err);
      alert('No se pudo comunicar con el backend local de Gemini. Usando Mockup de respaldo.');
      
      // Respaldo Mock local en caso de error
      setPaperForm(prev => ({
        ...prev,
        description: "Este artículo de investigación presenta un análisis del estado del arte en la aplicación de modelos avanzados de Inteligencia Artificial para el procesamiento de información multimodal de baja latencia.",
        imageUrl: "AI Suggested prompt: A futuristic holographic visual database connecting image, sound, and text --ar 16:9"
      }));
    } finally {
      setAnalyzingPaper(false);
    }
  };

  // Guardar Noticia
  const handleSaveNoticia = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!noticiaForm.title || !noticiaForm.content) {
      setErrorMsg('El título y contenido de la noticia son requeridos.');
      return;
    }

    setSaving(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    playWaterDrip();

    try {
      let finalImageUrl = noticiaForm.imageUrl;
      if (noticiaFile) {
        finalImageUrl = await uploadToStorage(noticiaFile, 'noticias');
      }

      const formattedDate = noticiaForm.date || new Date().toLocaleDateString('es-ES', { month: 'short', year: 'numeric' });

      if (isMockMode) {
        // Mock Save
        console.log("Mock saved news:", { ...noticiaForm, date: formattedDate, image_url: finalImageUrl });
        setSuccessMsg('¡Noticia simulada creada correctamente! (Modo Local).');
        // Reset
        setNoticiaForm({ title: '', content: '', tag: 'Novedad', date: '', imageUrl: '' });
        setNoticiaFile(null);
      } else {
        // Real Supabase Insert
        const { error } = await supabase
          .from('noticias')
          .insert({
            title: noticiaForm.title,
            content: noticiaForm.content,
            tag: noticiaForm.tag,
            date: formattedDate,
            image_url: finalImageUrl || null
          });

        if (error) throw error;
        setSuccessMsg('¡Noticia publicada exitosamente en la base de datos de Supabase!');
        setNoticiaForm({ title: '', content: '', tag: 'Novedad', date: '', imageUrl: '' });
        setNoticiaFile(null);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error al guardar la noticia.');
    } finally {
      setSaving(false);
    }
  };

  // Guardar Paper
  const handleSavePaper = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!paperForm.title || !paperForm.description || !paperForm.category) {
      setErrorMsg('Título, descripción y categoría son requeridos.');
      return;
    }

    setSaving(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    playWaterDrip();

    try {
      let finalPdfUrl = paperForm.url;
      if (paperFile) {
        finalPdfUrl = await uploadToStorage(paperFile, 'papers');
      }

      const formattedDate = paperForm.date || new Date().toLocaleDateString('es-ES', { month: 'long', year: 'numeric' });

      if (isMockMode) {
        // Mock Save
        console.log("Mock saved paper:", { 
          ...paperForm, 
          date: formattedDate, 
          url: finalPdfUrl,
          authors: selectedAuthors 
        });
        setSuccessMsg('¡Paper simulado guardado correctamente! (Modo Local).');
        setPaperForm({ title: '', description: '', category: 'Inteligencia Artificial', date: '', url: '', imageUrl: '', paperText: '' });
        setPaperFile(null);
        setSelectedAuthors([]);
      } else {
        // Real Supabase Insert
        // 1. Insertar en tabla de papers
        const { data: insertedPaper, error: paperError } = await supabase
          .from('papers')
          .insert({
            title: paperForm.title,
            description: paperForm.description,
            category: paperForm.category,
            date: formattedDate,
            url: finalPdfUrl,
            image_url: paperForm.imageUrl.startsWith('AI') ? null : (paperForm.imageUrl || null)
          })
          .select()
          .single();

        if (paperError) throw paperError;

        // 2. Insertar relaciones M-M de autores en paper_authors
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

        setSuccessMsg('¡Paper académico e indexación de autores guardados con éxito!');
        setPaperForm({ title: '', description: '', category: 'Inteligencia Artificial', date: '', url: '', imageUrl: '', paperText: '' });
        setPaperFile(null);
        setSelectedAuthors([]);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error al guardar el paper académico.');
    } finally {
      setSaving(false);
    }
  };

  const toggleAuthorSelection = (authorId: string) => {
    playRubberSnap();
    if (selectedAuthors.includes(authorId)) {
      setSelectedAuthors(prev => prev.filter(id => id !== authorId));
    } else {
      setSelectedAuthors(prev => [...prev, authorId]);
    }
  };

  // Cargar visuales del Dashboard/Login
  if (authLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] text-text-primary px-6">
        <RefreshCw className="w-10 h-10 text-primary-container animate-spin mb-4" />
        <p className="font-bold tracking-widest text-xs uppercase text-text-secondary">Autenticando credenciales...</p>
      </div>
    );
  }

  return (
    <div className="admin-page-container pt-28 pb-20 px-4 md:px-8 max-w-6xl mx-auto min-h-screen text-text-primary">
      
      {/* Mensajes Flotantes de Estado */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div 
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            className="fixed top-24 left-1/2 -translate-x-1/2 z-50 bg-[#261016] border border-[#FF3344] text-[#FF99AA] px-6 py-3 rounded-full flex items-center gap-3 text-sm font-semibold uppercase tracking-wider"
          >
            <AlertCircle className="w-4 h-4 shrink-0 text-[#FF3344]" />
            <span>{errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} className="ml-4 hover:opacity-75">✕</button>
          </motion.div>
        )}

        {successMsg && (
          <motion.div 
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            className="fixed top-24 left-1/2 -translate-x-1/2 z-50 bg-[#102416] border border-[#33FF44] text-[#99FFAA] px-6 py-3 rounded-full flex items-center gap-3 text-sm font-semibold uppercase tracking-wider"
          >
            <CheckCircle2 className="w-4 h-4 shrink-0 text-[#33FF44]" />
            <span>{successMsg}</span>
            <button onClick={() => setSuccessMsg(null)} className="ml-4 hover:opacity-75">✕</button>
          </motion.div>
        )}
      </AnimatePresence>

      {!session ? (
        /* VISTA 1: INICIO DE SESIÓN */
        <motion.div 
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-md mx-auto bg-surface-deep border border-border-organic rounded-[2.5rem] p-8 md:p-10 relative overflow-hidden mt-6"
        >
          {/* Indicador de modo desarrollador local */}
          {isMockMode && (
            <div className="absolute top-0 left-0 right-0 bg-[#FF5500]/10 border-b border-[#FF5500]/20 text-[#FF8c00] text-[10px] font-black text-center py-1.5 uppercase tracking-widest flex items-center justify-center gap-1.5 px-4">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Modo Desarrollador Local Activado</span>
            </div>
          )}

          <div className="flex flex-col items-center text-center mt-4 mb-8">
            <div className="w-16 h-16 bg-[#1A1124] rounded-full border border-border-organic flex items-center justify-center mb-4">
              <Lock className="w-6 h-6 text-primary-container" />
            </div>
            <h1 className="text-2xl font-black uppercase tracking-wider text-text-primary">GIBD CMS Portal</h1>
            <p className="text-text-secondary text-sm font-medium mt-2">Acceso administrativo para investigadores del grupo</p>
          </div>

          <form onSubmit={handleEmailLogin} className="flex flex-col gap-5">
            <div className="relative">
              <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
              <input 
                type="email" 
                placeholder="Email Institucional"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-4 pl-12 pr-6 text-sm font-medium focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/50"
              />
            </div>

            <div className="relative">
              <Key className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
              <input 
                type="password" 
                placeholder="Contraseña"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-4 pl-12 pr-6 text-sm font-medium focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/50"
              />
            </div>

            <motion.button 
              type="submit"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full bg-gradient-to-r from-primary-container to-[#ff8c00] text-white py-4 rounded-full font-black text-sm uppercase tracking-widest hover:shadow-[0_0_20px_rgba(255,85,0,0.2)] transition-all duration-300 mt-2"
            >
              Autenticar Administrador
            </motion.button>
          </form>

          <div className="relative my-6 flex items-center justify-center">
            <div className="absolute w-full h-[1px] bg-border-organic"></div>
            <span className="relative bg-surface-deep px-4 text-[10px] font-black text-text-secondary uppercase tracking-widest">O ACCEDER CON</span>
          </div>

          <motion.button 
            onClick={handleGoogleLogin}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary py-4 rounded-full font-bold text-sm uppercase tracking-wider flex items-center justify-center gap-3 hover:bg-surface-deep/40 transition-colors"
          >
            <svg className="w-4 h-4 text-primary-container fill-current shrink-0" viewBox="0 0 24 24">
              <path d="M12.24 10.285V13.4h6.887c-.648 2.41-2.519 4.113-5.136 4.113-3.18 0-5.762-2.573-5.762-5.748s2.58-5.748 5.762-5.748c1.497 0 2.842.543 3.9 1.4l2.42-2.42C18.6 3.4 15.66 2 12.24 2 6.58 2 2 6.58 2 12.24s4.58 10.24 10.24 10.24c5.795 0 10.24-4.11 10.24-10.24 0-.685-.082-1.354-.24-1.955H12.24z"/>
            </svg>
            Google Sign-in
          </motion.button>

          {isMockMode && (
            <div className="mt-6 p-4 rounded-2xl bg-[#0A0A0A] border border-border-organic text-left">
              <p className="text-[10px] font-black text-primary-container uppercase tracking-widest mb-1.5">Credenciales de Prueba:</p>
              <code className="text-xs text-text-secondary block">Email: admin@gibd.utn.edu.ar</code>
              <code className="text-xs text-text-secondary block">Pass: admin123</code>
            </div>
          )}
        </motion.div>
      ) : (
        /* VISTA 2: PANEL DE CONTROL DE ADMINISTRADOR */
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col gap-8"
        >
          {/* Header del Dashboard */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-surface-deep border border-border-organic rounded-[2rem] p-6 md:p-8">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-primary-container/10 border border-primary-container/20 rounded-full flex items-center justify-center">
                <UserCheck className="w-6 h-6 text-primary-container" />
              </div>
              <div>
                <span className="text-[10px] font-black text-primary-container uppercase tracking-widest">Panel del Investigador</span>
                <h2 className="text-lg md:text-xl font-black text-text-primary truncate max-w-xs md:max-w-md">
                  {session.user?.user_metadata?.full_name || session.user?.email}
                </h2>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {isMockMode && (
                <span className="bg-[#FF5500]/10 border border-[#FF5500]/30 text-primary-container text-[10px] font-black uppercase tracking-widest px-3 py-1.5 rounded-full">
                  Mock Mode
                </span>
              )}
              <motion.button 
                onClick={handleLogout}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                className="bg-[#0A0A0A] border border-border-organic text-text-primary hover:border-[#FF3344] hover:text-[#FF8899] px-5 py-2.5 rounded-full font-bold text-xs uppercase tracking-wider flex items-center gap-2 transition-all duration-300"
              >
                <LogOut className="w-3.5 h-3.5" />
                Cerrar Sesión
              </motion.button>
            </div>
          </div>

          {/* Menú de Navegación de Pestañas */}
          <div className="flex items-center p-1.5 bg-surface-deep rounded-full border border-border-organic w-full md:max-w-md mx-auto relative gap-1">
            {[
              { id: 'noticias', icon: Globe, label: "Noticias" },
              { id: 'papers', icon: BookOpen, label: "Papers" },
              { id: 'monitoreo', icon: Cpu, label: "Monitoreo" }
            ].map((tab) => (
              <button 
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id as any);
                  playWaterDrip();
                }}
                className="relative flex-1 py-3 flex items-center justify-center gap-2 rounded-full font-black text-xs uppercase tracking-widest transition-all duration-300 select-none z-10 text-center"
              >
                {activeTab === tab.id && (
                  <motion.div
                    layoutId="admin-active-tab-pill"
                    className="absolute inset-0 bg-[#0A0A0A] border border-border-organic rounded-full z-0 pointer-events-none"
                    transition={{ type: "spring", stiffness: 350, damping: 25 }}
                  />
                )}
                
                <tab.icon className={`w-4 h-4 relative z-10 transition-all duration-300 ${
                  activeTab === tab.id ? 'text-primary-container scale-110' : 'text-text-secondary'
                }`} />
                
                <span className={`relative z-10 transition-all duration-300 ${
                  activeTab === tab.id ? 'text-primary-container font-black' : 'text-text-secondary font-medium'
                }`}>
                  {tab.label}
                </span>
              </button>
            ))}
          </div>

          {/* Contenido Dinámico de las Pestañas */}
          <div className="admin-tab-content-container">
            <AnimatePresence mode="wait">
              {activeTab === 'noticias' && (
                /* SECCIÓN 1: NOTICIAS CMS */
                <motion.div 
                  key="noticias-tab"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.2 }}
                  className="bg-surface-deep border border-border-organic rounded-[2.5rem] p-6 md:p-8"
                >
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 bg-primary-container/10 border border-primary-container/20 rounded-full flex items-center justify-center">
                      <Globe className="w-5 h-5 text-primary-container" />
                    </div>
                    <div>
                      <h3 className="text-xl font-black uppercase tracking-wider text-text-primary">Publicar Nueva Noticia</h3>
                      <p className="text-text-secondary text-xs font-semibold">Crea novedades que aparecerán en la sección principal del GIBD</p>
                    </div>
                  </div>

                  <form onSubmit={handleSaveNoticia} className="flex flex-col gap-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="flex flex-col gap-2">
                        <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Título de la Noticia</label>
                        <input 
                          type="text" 
                          placeholder="Ej: Aceptación en CoNaIISI 2024 de NLP..."
                          value={noticiaForm.title}
                          onChange={(e) => setNoticiaForm(prev => ({ ...prev, title: e.target.value }))}
                          className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/40"
                          required
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Etiqueta / Tag</label>
                          <select 
                            value={noticiaForm.tag}
                            onChange={(e) => setNoticiaForm(prev => ({ ...prev, tag: e.target.value }))}
                            className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors appearance-none cursor-pointer"
                          >
                            <option value="Novedad">Novedad</option>
                            <option value="Publicación">Publicación</option>
                            <option value="Investigación">Investigación</option>
                            <option value="Conferencia">Conferencia</option>
                            <option value="Académico">Académico</option>
                          </select>
                        </div>

                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Fecha Publicación (Opcional)</label>
                          <input 
                            type="text" 
                            placeholder="Ej: Nov 2024"
                            value={noticiaForm.date}
                            onChange={(e) => setNoticiaForm(prev => ({ ...prev, date: e.target.value }))}
                            className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/40"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Contenido de la Novedad</label>
                      <textarea 
                        rows={4}
                        placeholder="Escribe el cuerpo de la noticia. Sé conciso y claro con las novedades del equipo."
                        value={noticiaForm.content}
                        onChange={(e) => setNoticiaForm(prev => ({ ...prev, content: e.target.value }))}
                        className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-[1.5rem] py-4 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/40 resize-y"
                        required
                      />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Subida de Imagen por Drag & Drop */}
                      <div className="flex flex-col gap-2">
                        <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Imagen Ilustrativa</label>
                        <div 
                          onDragEnter={(e) => handleDrag(e, 'noticia')}
                          onDragLeave={(e) => handleDrag(e, 'noticia')}
                          onDragOver={(e) => handleDrag(e, 'noticia')}
                          onDrop={(e) => handleDrop(e, 'noticia')}
                          className={`dragzone-container flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-[1.5rem] text-center transition-all ${
                            noticiaDragActive ? 'border-primary-container bg-primary-container/5' : 'border-border-organic hover:border-border-organic/80'
                          }`}
                        >
                          <Upload className="w-8 h-8 text-primary-container mb-3" />
                          {noticiaFile ? (
                            <div>
                              <p className="text-xs font-bold text-text-primary truncate max-w-[250px]">{noticiaFile.name}</p>
                              <p className="text-[10px] text-text-secondary mt-1">{(noticiaFile.size / 1024 / 1024).toFixed(2)} MB</p>
                              <button 
                                type="button" 
                                onClick={() => setNoticiaFile(null)}
                                className="text-[10px] text-[#FF3344] font-black uppercase tracking-widest mt-2 hover:underline"
                              >
                                Quitar Imagen
                              </button>
                            </div>
                          ) : (
                            <>
                              <p className="text-xs font-semibold text-text-primary">Arrastra una imagen aquí o</p>
                              <label className="text-xs text-primary-container font-black uppercase tracking-widest cursor-pointer mt-1.5 hover:underline">
                                Selecciona un archivo
                                <input 
                                  type="file" 
                                  accept="image/*"
                                  className="hidden" 
                                  onChange={(e) => handleFileChange(e, 'noticia')}
                                />
                              </label>
                              <p className="text-[9px] text-text-secondary mt-2">Formatos: JPG, PNG, WEBP (Máx 5MB)</p>
                            </>
                          )}
                        </div>
                      </div>

                      {/* URL Alternativa */}
                      <div className="flex flex-col gap-2 justify-end">
                        <label className="text-xs font-black uppercase tracking-widest text-text-secondary">O Dirección URL de Imagen Alternativa</label>
                        <input 
                          type="url" 
                          placeholder="https://ejemplo.com/imagen.jpg"
                          value={noticiaForm.imageUrl}
                          onChange={(e) => setNoticiaForm(prev => ({ ...prev, imageUrl: e.target.value }))}
                          disabled={noticiaFile !== null}
                          className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors disabled:opacity-50 placeholder:text-text-secondary/40"
                        />
                        <p className="text-[10px] text-text-secondary font-medium leading-normal mt-1">
                          Nota: Si subes un archivo, este tendrá prioridad sobre la dirección URL.
                        </p>
                      </div>
                    </div>

                    <div className="flex justify-end mt-4">
                      <motion.button 
                        type="submit"
                        disabled={saving}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className="bg-gradient-to-r from-primary-container to-[#ff8c00] text-white px-10 py-4 rounded-full font-black text-sm uppercase tracking-widest flex items-center gap-2 hover:shadow-[0_0_20px_rgba(255,85,0,0.2)] disabled:opacity-50"
                      >
                        {saving ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            Guardando...
                          </>
                        ) : (
                          <>
                            <FileUp className="w-4 h-4" />
                            Guardar Noticia
                          </>
                        )}
                      </motion.button>
                    </div>
                  </form>
                </motion.div>
              )}

              {activeTab === 'papers' && (
                /* SECCIÓN 2: PAPERS CMS */
                <motion.div 
                  key="papers-tab"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.2 }}
                  className="bg-surface-deep border border-border-organic rounded-[2.5rem] p-6 md:p-8 flex flex-col gap-8"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-primary-container/10 border border-primary-container/20 rounded-full flex items-center justify-center">
                      <BookOpen className="w-5 h-5 text-primary-container" />
                    </div>
                    <div>
                      <h3 className="text-xl font-black uppercase tracking-wider text-text-primary">Indexar Paper Científico</h3>
                      <p className="text-text-secondary text-xs font-semibold">Agrega un artículo académico y asócialo a sus autores investigadores</p>
                    </div>
                  </div>

                  {/* Asistente AI integrado con Gemini */}
                  <div className="ai-assistant-card bg-[#1A1124] border border-border-organic rounded-[2rem] p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-24 h-24 bg-primary-container/5 rounded-full blur-2xl"></div>
                    <div className="flex items-center gap-2 mb-4">
                      <Sparkles className="w-4 h-4 text-primary-container" />
                      <h4 className="text-sm font-black uppercase tracking-widest text-primary-container">Optimizador Asistido por IA (Gemini)</h4>
                    </div>
                    
                    <p className="text-text-secondary text-xs font-medium leading-relaxed mb-4">
                      ¿Tienes el texto o abstract crudo del artículo? Pégalo debajo y deja que Gemini analice el paper para auto-redactar un resumen optimizado para la web y recomendar un prompt visual premium.
                    </p>

                    <div className="flex flex-col gap-3">
                      <textarea 
                        rows={3}
                        placeholder="Pega texto del paper aquí para optimizar con IA (Ej: Título, Abstract original, o introducción)..."
                        value={paperForm.paperText}
                        onChange={(e) => setPaperForm(prev => ({ ...prev, paperText: e.target.value }))}
                        className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-[1rem] py-3.5 px-5 text-xs font-medium focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/30 resize-y"
                      />
                      <div className="flex justify-between items-center gap-4">
                        <span className="text-[9px] text-text-secondary flex items-center gap-1 font-semibold">
                          <Activity className="w-3 h-3 text-primary-container" />
                          Requiere GIBD API Gateway levantado en local
                        </span>
                        <motion.button 
                          type="button"
                          onClick={optimizeWithGemini}
                          disabled={analyzingPaper}
                          whileHover={{ scale: 1.03 }}
                          whileTap={{ scale: 0.97 }}
                          className="bg-[#0A0A0A] border border-border-organic text-primary-container hover:border-primary-container px-6 py-2.5 rounded-full font-black text-[10px] uppercase tracking-widest flex items-center gap-2 disabled:opacity-50 transition-colors"
                        >
                          {analyzingPaper ? (
                            <>
                              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                              Analizando Paper...
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-3.5 h-3.5" />
                              Optimizar con Gemini
                            </>
                          )}
                        </motion.button>
                      </div>
                    </div>
                  </div>

                  <form onSubmit={handleSavePaper} className="flex flex-col gap-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="flex flex-col gap-2">
                        <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Título del Paper</label>
                        <input 
                          type="text" 
                          placeholder="Ej: Advanced Variable Tuning and Biases..."
                          value={paperForm.title}
                          onChange={(e) => setPaperForm(prev => ({ ...prev, title: e.target.value }))}
                          className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/40"
                          required
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Categoría Científica</label>
                          <input 
                            type="text" 
                            placeholder="Ej: Big Data, NLP, Metric Learning"
                            value={paperForm.category}
                            onChange={(e) => setPaperForm(prev => ({ ...prev, category: e.target.value }))}
                            className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/40"
                            required
                          />
                        </div>

                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Fecha y Evento</label>
                          <input 
                            type="text" 
                            placeholder="Ej: Oct 2024 (ARGENCON 2024)"
                            value={paperForm.date}
                            onChange={(e) => setPaperForm(prev => ({ ...prev, date: e.target.value }))}
                            className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/40"
                            required
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Resumen / Descripción Ejecutiva</label>
                      <textarea 
                        rows={4}
                        placeholder="Redacta el resumen que se mostrará en el catálogo web del GIBD."
                        value={paperForm.description}
                        onChange={(e) => setPaperForm(prev => ({ ...prev, description: e.target.value }))}
                        className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-[1.5rem] py-4 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/40 resize-y"
                        required
                      />
                    </div>

                    {/* Selector de Autores */}
                    <div className="flex flex-col gap-2">
                      <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Co-Autores Integrantes del GIBD</label>
                      {loadingAuthors ? (
                        <div className="py-4 text-xs text-text-secondary font-semibold">Cargando investigadores...</div>
                      ) : (
                        <div className="flex flex-wrap gap-2.5 mt-1.5">
                          {authors.map((author) => {
                            const isSelected = selectedAuthors.includes(author.id);
                            return (
                              <button
                                type="button"
                                key={author.id}
                                onClick={() => toggleAuthorSelection(author.id)}
                                className={`author-pill-selection px-5 py-2.5 rounded-full font-bold text-xs uppercase tracking-wider transition-all duration-200 flex items-center gap-2 border select-none ${
                                  isSelected 
                                    ? 'bg-primary-container/10 border-primary-container text-primary-container' 
                                    : 'bg-[#0A0A0A] border-border-organic text-text-secondary hover:border-border-organic/80 hover:text-text-primary'
                                }`}
                              >
                                <span className="initials-badge w-6 h-6 bg-surface-deep border border-border-organic rounded-full flex items-center justify-center text-[10px] text-text-primary font-black">
                                  {author.initials}
                                </span>
                                <span>{author.name}</span>
                                {isSelected && <Check className="w-3.5 h-3.5 ml-1 shrink-0" />}
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-2">
                      {/* Subida del PDF del Paper */}
                      <div className="flex flex-col gap-2">
                        <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Documento Académico (PDF)</label>
                        <div 
                          onDragEnter={(e) => handleDrag(e, 'paper')}
                          onDragLeave={(e) => handleDrag(e, 'paper')}
                          onDragOver={(e) => handleDrag(e, 'paper')}
                          onDrop={(e) => handleDrop(e, 'paper')}
                          className={`dragzone-container flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-[1.5rem] text-center transition-all ${
                            paperDragActive ? 'border-primary-container bg-primary-container/5' : 'border-border-organic hover:border-border-organic/80'
                          }`}
                        >
                          <Upload className="w-8 h-8 text-primary-container mb-3" />
                          {paperFile ? (
                            <div>
                              <p className="text-xs font-bold text-text-primary truncate max-w-[250px]">{paperFile.name}</p>
                              <p className="text-[10px] text-text-secondary mt-1">{(paperFile.size / 1024 / 1024).toFixed(2)} MB</p>
                              <button 
                                type="button" 
                                onClick={() => setPaperFile(null)}
                                className="text-[10px] text-[#FF3344] font-black uppercase tracking-widest mt-2 hover:underline"
                              >
                                Quitar Documento
                              </button>
                            </div>
                          ) : (
                            <>
                              <p className="text-xs font-semibold text-text-primary">Arrastra el archivo PDF aquí o</p>
                              <label className="text-xs text-primary-container font-black uppercase tracking-widest cursor-pointer mt-1.5 hover:underline">
                                Selecciona un archivo
                                <input 
                                  type="file" 
                                  accept="application/pdf"
                                  className="hidden" 
                                  onChange={(e) => handleFileChange(e, 'paper')}
                                />
                              </label>
                              <p className="text-[9px] text-text-secondary mt-2">Formato: PDF Científico (Máx 25MB)</p>
                            </>
                          )}
                        </div>
                      </div>

                      {/* URL PDF Alternativa e Imagen */}
                      <div className="flex flex-col gap-4">
                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-black uppercase tracking-widest text-text-secondary">O Enlace de Descarga Directa (URL)</label>
                          <input 
                            type="url" 
                            placeholder="https://ejemplo.com/paper.pdf"
                            value={paperForm.url}
                            onChange={(e) => setPaperForm(prev => ({ ...prev, url: e.target.value }))}
                            disabled={paperFile !== null}
                            className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors disabled:opacity-50 placeholder:text-text-secondary/40"
                          />
                        </div>

                        <div className="flex flex-col gap-2">
                          <label className="text-xs font-black uppercase tracking-widest text-text-secondary">Dirección URL de Imagen de Portada (Opcional)</label>
                          <input 
                            type="text" 
                            placeholder="Sugerido por IA o enlace directo..."
                            value={paperForm.imageUrl}
                            onChange={(e) => setPaperForm(prev => ({ ...prev, imageUrl: e.target.value }))}
                            className="w-full bg-[#0A0A0A] border border-border-organic text-text-primary rounded-full py-3.5 px-6 text-sm font-semibold focus:outline-none focus:border-primary-container transition-colors placeholder:text-text-secondary/40"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex justify-end mt-4">
                      <motion.button 
                        type="submit"
                        disabled={saving}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className="bg-gradient-to-r from-primary-container to-[#ff8c00] text-white px-10 py-4 rounded-full font-black text-sm uppercase tracking-widest flex items-center gap-2 hover:shadow-[0_0_20px_rgba(255,85,0,0.2)] disabled:opacity-50"
                      >
                        {saving ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            Guardando...
                          </>
                        ) : (
                          <>
                            <FileUp className="w-4 h-4" />
                            Indexar e Inserter Paper
                          </>
                        )}
                      </motion.button>
                    </div>
                  </form>
                </motion.div>
              )}

              {activeTab === 'monitoreo' && (
                /* SECCIÓN 3: MONITOR DE ESTADO Y DIAGNÓSTICO */
                <motion.div 
                  key="monitoreo-tab"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.2 }}
                  className="bg-surface-deep border border-border-organic rounded-[2.5rem] p-6 md:p-8 flex flex-col gap-6"
                >
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-primary-container/10 border border-primary-container/20 rounded-full flex items-center justify-center">
                        <Cpu className="w-5 h-5 text-primary-container" />
                      </div>
                      <div>
                        <h3 className="text-xl font-black uppercase tracking-wider text-text-primary">Servicio de Diagnósticos</h3>
                        <p className="text-text-secondary text-xs font-semibold">Estado de la pasarela local de IA y conexiones relacionales</p>
                      </div>
                    </div>
                    
                    <motion.button 
                      onClick={checkBackendHealth}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="p-2.5 bg-[#0A0A0A] border border-border-organic rounded-full hover:border-primary-container text-text-secondary hover:text-primary-container transition-colors"
                      title="Refrescar Estado"
                    >
                      <RefreshCw className={`w-4 h-4 ${backendHealth.loading ? 'animate-spin' : ''}`} />
                    </motion.button>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Diagnóstico 1: Gateway backend */}
                    <div className="bg-[#0A0A0A] border border-border-organic p-6 rounded-[2rem] flex flex-col justify-between min-h-[140px]">
                      <span className="text-[10px] font-black uppercase tracking-widest text-text-secondary">AI API Gateway</span>
                      <div className="flex items-center gap-2.5 my-3">
                        {backendHealth.error ? (
                          <XCircle className="w-5 h-5 text-[#FF3344] shrink-0" />
                        ) : (
                          <CheckCircle2 className="w-5 h-5 text-[#33FF44] shrink-0" />
                        )}
                        <span className="font-bold text-sm">FastAPI Backend</span>
                      </div>
                      <span className={`text-[10px] font-black uppercase tracking-widest ${backendHealth.error ? 'text-[#FF3344]' : 'text-[#33FF44]'}`}>
                        {backendHealth.error ? 'INACCESIBLE' : 'ACTIVO (PUERTO 8000)'}
                      </span>
                    </div>

                    {/* Diagnóstico 2: Gemini SDK */}
                    <div className="bg-[#0A0A0A] border border-border-organic p-6 rounded-[2rem] flex flex-col justify-between min-h-[140px]">
                      <span className="text-[10px] font-black uppercase tracking-widest text-text-secondary">Google AI Inferencia</span>
                      <div className="flex items-center gap-2.5 my-3">
                        <Sparkles className={`w-5 h-5 text-primary-container ${backendHealth.gemini_active ? 'glow-icon-orange' : 'opacity-40'} shrink-0`} />
                        <span className="font-bold text-sm">Gemini-1.5-Flash</span>
                      </div>
                      <span className={`text-[10px] font-black uppercase tracking-widest ${backendHealth.gemini_active ? 'text-[#33FF44]' : 'text-text-secondary'}`}>
                        {backendHealth.gemini_active ? 'API KEY CONECTADA' : 'MOCK DE RESPALDO'}
                      </span>
                    </div>

                    {/* Diagnóstico 3: Base de Datos Relacional */}
                    <div className="bg-[#0A0A0A] border border-border-organic p-6 rounded-[2rem] flex flex-col justify-between min-h-[140px]">
                      <span className="text-[10px] font-black uppercase tracking-widest text-text-secondary">Persistencia Relacional</span>
                      <div className="flex items-center gap-2.5 my-3">
                        <Database className="w-5 h-5 text-primary-container shrink-0" />
                        <span className="font-bold text-sm">PostgreSQL DB</span>
                      </div>
                      <span className={`text-[10px] font-black uppercase tracking-widest ${isMockMode ? 'text-text-secondary' : 'text-[#33FF44]'}`}>
                        {isMockMode ? 'MOCK LOCAL (SQLITE/MEM)' : 'CONECTADO A SUPABASE'}
                      </span>
                    </div>
                  </div>

                  {/* Detalle de Variables de Entorno del Cliente */}
                  <div className="bg-[#0A0A0A] border border-border-organic p-6 rounded-[2rem] mt-2">
                    <h4 className="text-xs font-black uppercase tracking-widest text-primary-container mb-4">Variables de Entorno del Cliente</h4>
                    <div className="flex flex-col gap-3 font-semibold text-xs text-text-secondary">
                      <div className="flex justify-between border-b border-border-organic/40 pb-2">
                        <span>VITE_SUPABASE_URL</span>
                        <code className="text-text-primary text-[10px] truncate max-w-[200px] md:max-w-md">
                          {import.meta.env.VITE_SUPABASE_URL || 'No Configurada / Usando Mock'}
                        </code>
                      </div>
                      <div className="flex justify-between pt-1">
                        <span>VITE_SUPABASE_ANON_KEY</span>
                        <code className="text-text-primary text-[10px] truncate max-w-[200px] md:max-w-md">
                          {import.meta.env.VITE_SUPABASE_ANON_KEY ? '✔ Configurada (Oculta)' : 'No Configurada / Usando Mock'}
                        </code>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      )}

    </div>
  );
};

export default Admin;
