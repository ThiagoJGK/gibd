# Gemini & Antigravity Project Instructions — GIBD 2026

Estas instrucciones aplican de forma inmediata y automática para cualquier sesión de Antigravity / Gemini en este repositorio:

1. **Liderazgo:** Thiago Gomez Kehler es Tech Lead, Arquitecto de Software y Diseñador UI/UX Principal. Acatar sus lineamientos técnicos y de diseño.
2. **Central de Planificación:** Consultar siempre [`docs/plan_maestro_tech_lead_thiago.md`](docs/plan_maestro_tech_lead_thiago.md) y [`docs/semanas/`](docs/semanas/) antes de proponer cambios arquitecturales o de cronograma.
3. **Autenticidad:** NUNCA simular datos con hashes aleatorios o MD5. Usar el catálogo real de 80 marcas de ganado del GIBD en `backend/app/data/brands/` y `frontend/public/brands/`.
4. **Flat Design System:**
   - Botones, inputs, badges y selectores en píldora (`rounded-full` / `border-radius: 9999px`).
   - Paneles y tarjetas en radio orgánico (`rounded-[2rem]`).
   - Fondo OLED `#0A0A0A`, acento `#FF5500`, cero `box-shadow` difuso.
   - Microinteracciones suaves con `framer-motion`.
5. **Aislamiento:** No ensuciar componentes de producción con URLs estáticas a puertos locales (`localhost:8010`, etc.).
6. **Compilación:** Verificar siempre que `npm run build` en `frontend/` termine con código 0 antes de dar una tarea por completada.
7. **Regla de Git:** Luego de grandes cambios, preguntar siempre si continuar con subida a Git.
