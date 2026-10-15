# 📋 Plan Técnico: Semana 2 — Emmanuel Davezac (Frontend & CMS)

> **Rol:** Frontend & CMS Admin  
> **Ciclo:** Sprint 2  
> **Rama de Trabajo:** `feature/semana2-frontend-cms` (creada a partir de `dev`)  
> **Fecha de Entrega:** Viernes (Puesta en Común)  

---

## 🎯 Objetivo
Construir los formularios de administración interactivos en `Admin.tsx` para permitir a los investigadores publicar Noticias institucionales y Papers académicos con archivo PDF y multi-autoría, respetando de manera estricta el Flat Design System.

---

## 🎨 Especificación Estética: Flat Design System (Obligatorio)

* **Geometría de Cápsula:** Botones, inputs, badges, selectores y chips de autor con `rounded-full` (`border-radius: 9999px`).
* **Contenedores y Paneles:** Tarjetas y secciones con radio orgánico `rounded-[2rem]`.
* **Paleta:**
  * Fondo base: `#0A0A0A` (OLED Puro) / paneles `card-glass-purple`.
  * Acento primario: `#FF5500` (Naranja eléctrico).
  * Textos: `#FFFFFF` (títulos y contenido clave), `#A1A1AA` (etiquetas y placeholders).
* **Cero Sombras Difusas:** Prohibido el uso de `box-shadow` difusos convencionales. La profundidad se delimita con microbordes orgánicos de alto contraste (`border border-white/10` o `border border-border-organic`).
* **Animaciones:** Usar `framer-motion` para transiciones de pestañas y estados de carga.

---

## 📑 Contratos de Datos (TypeScript en `frontend/src/utils/dbService.ts`)

```typescript
export interface Noticia {
  id: string;
  titulo: string;
  bajada: string;
  cuerpo: string;
  imagen_url: string;
  fecha: string;
}

export interface Paper {
  id: string;
  titulo: string;
  abstract: string;
  anio: number;
  categoria: 'IA' | 'Legal' | 'Agro' | 'General';
  pdf_url: string;
  autores_ids: string[]; // IDs seleccionados de la lista de 24 investigadores
}
```

---

## 🛠️ Tareas Técnicas Paso a Paso

### Tarea 1: Pestaña "Noticias e Hitos" en `Admin.tsx`
- [ ] Maquetar el formulario de alta de noticias:
  - Input de Título (`rounded-full`).
  - Input de Bajada corta (`rounded-full`).
  - Textarea de Cuerpo de la noticia (`rounded-[1.5rem]`).
  - Dropzone para imagen de portada con previsualización inmediata y botón de descarte.
- [ ] Conectar el envío con `dbService.createNoticia(...)` mostrando feedback de guardado exitoso.

### Tarea 2: Pestaña "Papers Académicos" en `Admin.tsx`
- [ ] Maquetar el formulario de alta de papers:
  - Input de Título y Año de publicación.
  - Dropdown o selector de Categoría (`'IA' | 'Legal' | 'Agro' | 'General'`).
  - Dropzone / input de archivo para subir el documento `.pdf`.
- [ ] **Selector Multi-Autor Interactivo:**
  - Renderizar a los **24 investigadores reales** del GIBD como chips/píldoras clickeables (`rounded-full`).
  - Permitir seleccionar múltiples coautores con toggle visual (borde naranja al seleccionar).
  - Garantizar que la lista de chips sea responsiva y no genere desbordamiento horizontal (`overflow-x`) en smartphones o tablets.

### Tarea 3: Botón de Asistencia de IA en el Abstract
- [ ] Junto al campo de Abstract, incluir el botón en cápsula: `"Autocompletar con IA"`.
- [ ] Al pulsarlo, enviar el texto cargado a `http://localhost:8000/api/v1/ai/analyze-paper`.
- [ ] Si la llamada es exitosa, autocompletar el campo de texto del abstract con el contenido devuelto por Gemini. Si falla la red, mostrar un mensaje de fallback amigable sin bloquear al usuario.

---

## 🛡️ Quality Gate (Criterios de Aprobación para Merge)
1. `npm run build` en la carpeta `frontend/` finaliza con **código 0** y sin ningún error de TypeScript.
2. Los formularios no contienen enlaces estáticos a puertos deprecados (`localhost:8010` o `localhost:9999`).
3. El diseño responde al 100% en pantallas móviles (375px) y de escritorio (1440px).
4. La información guardada persiste al recargar la página (F5) mediante el mock local de `dbService`.
