# Plan de Fase 3: CMS de Grupos de Investigación y Equipo de Trabajo

## 1. Visión General de la Fase
La Fase 3 extiende el panel de administración (`/admin`) y la Landing Page para incorporar la gestión integral y dinámica de los **Grupos de Investigación** (Sonido, Video, Texto, Imágenes, Reducción de Dimensionalidad, Aplicaciones) y del **Equipo de Trabajo (Investigadores, Docentes y Alumnos)**. Se elimina cualquier dato estático de la estructura de investigación y del staff de la universidad.

> [!IMPORTANT]
> **Objetivo No-Dev CMS:** Permitir crear y modificar grupos de trabajo, así como añadir nuevos integrantes cargando foto, nombre, email, perfil de LinkedIn y asignación de grupo directamente desde la interfaz web.

---

## 2. Distribución de Responsabilidades (3 Desarrolladores)

### **Desarrollador 1 (Thiago Gomez Kehler - Tech Lead & Data Architecture)**
- **Esquema de BD para Grupos y Equipo:** Diseñar e implementar la tabla `grupos_investigacion` y enriquecer `miembros_equipo`.
- **Extensión en dbService.ts:** Crear métodos `getGruposInvestigacion`, `saveGrupoInvestigacion`, `saveMiembroEquipo` y `deleteMiembroEquipo`.

### **Desarrollador 2 (Emanuel Davezac - Backend & Admin Components)**
- **Sección Admin de Grupos:** Construir la pestaña y formularios en `Admin.tsx` para la gestión CRUD de Grupos de Investigación.
- **Sección Admin de Integrantes:** Construir la pestaña y formularios en `Admin.tsx` para la gestión de integrantes con subida de fotos de avatar.

### **Desarrollador 3 (Renato Bonín - Frontend & UI Components)**
- **Landing Page - Sección Grupos:** Rediseñar dinámicamente la sección "Grupos de Investigación" en `Landing.tsx` para renderizar las tarjetas de cada grupo desde la BD/LocalStorage.
- **Componentes de Modal/Tarjetas de Equipo:** Implementar componentes de interfaz interactivos para explorar el equipo de trabajo por rol o subgrupo.

---

## 3. Tareas Detalladas de la Fase 3

- [ ] **Tarea 3.1: Actualización del Esquema SQL para Grupos y Equipo** (Dev 1)
- [ ] **Tarea 3.2: Métodos CRUD en dbService.ts** (Dev 1)
- [ ] **Tarea 3.3: Panel de Administración para Grupos y Miembros** (Dev 2)
- [ ] **Tarea 3.4: Renderizado Dinámico en Landing Page** (Dev 3)

---

## 4. Plan de Comprobación, Validación y Testing

### A. Pruebas Automáticas y Compilación
```powershell
cd frontend
npm run build
```

### B. Pruebas Manuales de Funcionamiento
1. Crear un nuevo grupo y un nuevo integrante en Admin.
2. Verificar la aparición inmediata y persistencia en la Landing Page.

---

## 5. Protocolo de Finalización y Subida a Git

1. **Peer Review:** Auditoría cruzada.
2. **Validación de Build:** `npm run build`
3. **Consulta de Subida (Regla Global):**
   - **PREGUNTAR AL USUARIO:** "La Fase 3 (CMS de Grupos y Equipo) ha sido completada y testeada con éxito. ¿Podemos proceder a realizar el push a Git de esta fase?"
4. **Git Merge & Push:** Tras confirmación explícita del usuario.
