# Plan de Subfase 1.5: Pruebas E2E de Fase 1, Validación Local y Consulta para Git Push

## 1. Información General
- **Fase Perteneciente:** FASE 1 - Core de IA de Marcas de Ganado, Gateway Python y Wrapper Embebido
- **Subfase:** 1.5
- **Desarrolladores Asignados:** Todo el equipo (Thiago Gomez Kehler, Emanuel Davezac, Renato Bonín)

---

## 2. Objetivo de la Subfase
Integrar las subfases 1.1, 1.2, 1.3 y 1.4 en una suite de validación punta a punta (End-to-End). Garantizar que la compilación de producción del frontend (`npm run build`) sea 100% limpia y que la comunicación con el backend en Python responda con alta precisión. Al certificar el funcionamiento, se realiza la consulta al usuario para autorizar la subida a Git.

---

## 3. Tareas Técnicas Detalladas

- [ ] **Tarea 1.5.1: Auditoría de Compilación de Producción**
  - Comando: `cd frontend; npm run build`
  - Asegurar cero errores de TypeScript y cero advertencias de bundles.
- [ ] **Tarea 1.5.2: Matriz de Integración E2E (Flujo Marcas de Ganado & Python Wrapper)**
  - Probar flujo completo: Carga Drag & Drop -> llamada FastAPI -> renderizado de tarjetas Top-K -> conmutación al marco embebido `PythonAppWrapper`.
- [ ] **Tarea 1.5.3: Consulta y Protocolo de Subida a Git (Regla Global)**
  - Al completar la verificación, **PREGUNTAR AL USUARIO:**
    *"La Fase 1 (Core de IA de Marcas de Ganado, Gateway Python y Python App Wrapper) ha sido probada y validada exitosamente en entorno local. ¿Deseas autorizar la subida (push) a Git de esta fase?"*
  - Tras recibir confirmación explícita:
    ```powershell
    git checkout main
    git merge feature/fase-1-ia-core
    git push origin main
    ```

---

## 4. Plan de Comprobación, Validación y Testing

### Matriz de Pruebas Integrales
1. **Compilación Frontend:** `npm run build` -> Éxito total.
2. **Backend FastAPI:** Inferencia activa en `http://localhost:8000`.
3. **Flujo de Usuario Completo:** 
   - Entrada a `/laboratorio`.
   - Subida de marca de ganado -> procesamiento -> grilla Top-K con métricas de similitud.
   - Conmutación a la vista de programa embebido de Python sin ruptura visual.

---

## 5. Criterios de Aceptación de la Subfase
- Toda la Fase 1 validada y funcionando de principio a fin.
- Código limpio sin advertencias de linter.
- Autorización explícita del usuario obtenida antes de ejecutar `git push`.
