---
trigger: always_on
description: Reglas de orden, calidad de datos y directrices estéticas para el proyecto GIBD WEB.
---

# Reglas de Orden y Disciplina Operativa — GIBD

1. **Liderazgo Técnico y Estético:** Thiago Gomez Kehler es el Tech Lead y Diseñador Principal. Garantizar siempre que el código y la interfaz cumplan con los estándares definidos en `docs/plan_maestro_tech_lead_thiago.md`.
2. **Prohibición de Datos Falsos:** NUNCA usar hashes MD5 ni generadores aleatorios para simular inferencias. Usar los 80 activos reales de marcas de ganado del GIBD disponibles en `frontend/public/brands/` y `backend/app/data/brands/`.
3. **Flat Design System:**
   - Geometría de cápsula (`rounded-full`) para todos los elementos interactivos.
   - Contenedores principales con radio orgánico (`rounded-[2rem]`).
   - Fondo OLED `#0A0A0A`, acento `#FF5500`.
   - Prohibido el uso de `box-shadow` difusos convencionales.
4. **Control de Versiones:** Prohibido tocar `main` directamente. Trabajar en `dev` y `feature/*`. Siempre consultar antes de subir cambios mayores a Git.
5. **Quality Gate:** Ejecutar y validar `npm run build` en `frontend/` sin errores de compilación ni TypeScript.
