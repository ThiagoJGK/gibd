# 📅 Plan de Trabajo: Semana 3 — CMS de los 6 Grupos de Investigación y Equipo

> **Ciclo:** Sprint 3  
> **Objetivo Central:** Estructurar dinámicamente las 6 líneas oficiales de investigación del GIBD (del Vault de Obsidian) y gestionar al equipo de docentes y becarios alumnos desde `/admin` reflejándolos en la Landing Page.  
> **Puesta en Común:** Viernes (Reunión Semanal del Equipo)  

---

## 1. Contratos de Integración (Definidos en el Kick-off del Lunes)

### Taxonomía Oficial de Grupos (Obsidian Vault MOC):
1. `G1` - Sonido (Audio, espectrogramas, aprendizaje métrico)
2. `G2` - Video (Búsqueda por similitud, detección de vehículos/patentes)
3. `G3` - Texto (Sentence embeddings, NLP, PTAH-Jurídico)
4. `G4` - Imágenes (Recuperación visual, Redes Siamesas, Marcas de Ganado)
5. `G5` - Reducción (PCA, UMAP, Autoencoders, representaciones compactas)
6. `G6` - Desarrollo (Plataforma Web, aplicaciones móviles, APIs)

### Contrato de Datos (TypeScript en `dbService.ts`):
```typescript
interface GrupoInvestigacion {
  id: string;
  codigo: 'G1' | 'G2' | 'G3' | 'G4' | 'G5' | 'G6';
  nombre: string;
  descripcion: string;
  icono: string;
  coordinador: string;
}

interface MiembroEquipo {
  id: string;
  nombre_completo: string;
  rol: 'Docente Investigador' | 'Becario Alumno' | 'Graduado Colaborador';
  email: string;
  avatar_url: string;
  linkedin_url?: string;
  grupo_codigo: 'G1' | 'G2' | 'G3' | 'G4' | 'G5' | 'G6';
}
```

---

## 2. Distribución de Tareas por Integrante

### 🐍 Renato Bonín (Backend & IA)
- [ ] **Tarea 3.1: Endpoints de Persistencia para Grupos y Miembros**
  - Archivo: `backend/app/main.py`.
  - Crear endpoints CRUD auxiliares o sincronización directa con Supabase para listar y actualizar grupos y miembros.
  - Asegurar serialización limpia de perfiles y validación de URLs de avatares.

---

### ⚛️ Emmanuel Davezac (Frontend & CMS)
- [ ] **Tarea 3.2: Pestaña de Gestión de Integrantes en `Admin.tsx`**
  - Archivo: `frontend/src/pages/Admin.tsx`.
  - Formulario para dar de alta/editar integrantes: nombre, selector de rol (docente/alumno), email, enlace a LinkedIn, selector de grupo (G1 a G6) y subida de foto avatar.
- [ ] **Tarea 3.3: Pestaña de Gestión de Grupos en `Admin.tsx`**
  - Vista para actualizar la descripción, tecnologías clave y coordinador de cada uno de los 6 grupos de investigación.

---

### 👑 Thiago Gomez Kehler (Tech Lead & Diseñador)
- [ ] **Tarea 3.4: Base de Datos Relacional y Seed de Integrantes**
  - Archivo: `Contexto/database_schema.sql`.
  - Crear tablas `grupos_investigacion` y `miembros_equipo` con claves foráneas e integridad referencial.
  - Cargar los datos semilla reales de los docentes y alumnos extraídos del Obsidian Vault.
- [ ] **Tarea 3.5: Rediseño Dinámico de la Sección Grupos en `Landing.tsx`**
  - Maquetar las tarjetas interactivas de los 6 grupos con estética Flat (`border-border-organic`, fondo morado oscuro, hover expansivo y microinteracciones de Framer Motion).
- [ ] **Tarea 3.6: Componente Modal de Perfil de Investigadores**
  - Diseñar el modal emergente con geometría de cápsula para consultar la trayectoria, foto y papers asociados a cada integrante del GIBD.

---

## 3. Pruebas y Criterios de Aceptación

### A. Verificación Intermedia (Miércoles)
1. **Prueba de Subida de Avatar:** Comprobar que al cargar un avatar en el formulario de Emma, la imagen se almacene y no rompa el radio circular de la previsualización.
2. **Prueba de Filtro:** Verificar que los integrantes se agrupen correctamente según su código de grupo (`G1` a `G6`).

### B. Verificación de Entrega (Viernes)
1. **Compilación Limpia:** `npm run build` con código 0.
2. **Flujo E2E de Gestión de Equipo:**
   - Crear un nuevo integrante en `/admin`.
   - Navegar a `Landing.tsx` y comprobar que aparece en su grupo con su foto y enlace de LinkedIn funcional.
   - Probar en resolución desktop y mobile (375px) asegurando cero desbordes.

---

## 4. Protocolo de Reprogramación
Si la subida de avatares o la grilla de grupos en la Landing requieren refinamiento de diseño o estilos, la tarea de maquetado se extiende al inicio de la Semana 4 sin bloquear el desarrollo de los endpoints de lenguaje natural.
