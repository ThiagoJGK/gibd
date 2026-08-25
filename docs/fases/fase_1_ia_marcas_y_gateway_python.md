# Plan de Fase 1: Core de IA de Marcas de Ganado, Gateway de Inferencia Python y Wrapper Embebido (Prioridad Andrés Pascal & Adrián Planas)

## 1. Visión General de la Fase
Esta primera fase atiende de forma prioritaria lo acordado con **Andrés Pascal y Adrián Planas**: implementar el motor principal de Inteligencia Artificial para la **Búsqueda por Similitud de Marcas de Ganado** y habilitar la infraestructura web necesaria para ejecutarlo y visualizarlo.

Para garantizar un avance metódico, la **FASE 1 se divide en 5 Subfases**, cada una con su plan individual en formato `.md` dentro de [`docs/fases/fase_1/`](file:///C:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/fases/fase_1/):

1. **[Subfase 1.1: FastAPI Gateway y Engine de Inferencia (Renato)](file:///C:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/fases/fase_1/subfase_1_1_fastapi_gateway_siamese_engine.md)**
2. **[Subfase 1.2: Componente PythonAppWrapper - Marco Embebido (Emanuel)](file:///C:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/fases/fase_1/subfase_1_2_python_app_wrapper.md)**
3. **[Subfase 1.3: Frontend Laboratorio - Integración Drag & Drop e Inferencia en Vivo (Thiago & Renato)](file:///C:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/fases/fase_1/subfase_1_3_laboratorio_drag_drop_integracion.md)**
4. **[Subfase 1.4: UI de Ajustes Top-K, Slider de Umbrales y Tarjetas de Métricas de Similitud (Thiago)](file:///C:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/fases/fase_1/subfase_1_4_ui_topk_slider_y_metricas.md)**
5. **[Subfase 1.5: Pruebas E2E de Fase 1, Validación Local y Consulta para Git Push (Todo el Equipo)](file:///C:/Users/thiag/Desktop/Files/Projects/GIBD%20WEB/docs/fases/fase_1/subfase_1_5_testing_e2e_y_git_push.md)**

---

## 2. Distribución Reasignada de Responsabilidades (3 Desarrolladores)

### **Desarrollador 1 (Renato Bonín - Python AI Engine & Gateway)**
- **Gateway FastAPI (Backend Python):** Implementar la API de inferencia en `backend/app/main.py` y `backend/app/services/siamese_engine.py`:
  - Endpoint `/api/v1/inference/siamese-brands` que recibe la imagen, ejecuta normalización y calcula la matriz de distancias métricas.
  - Retorno de JSON estandarizado con Top-K y distancias relativas.

### **Desarrollador 2 (Emanuel Davezac - Componente Wrapper)**
- **Componente PythonAppWrapper (`PythonAppWrapper.tsx`):** Construir el marco/wrapper adaptativo para integrar dashboards y programas embebidos de Python en la web.

### **Desarrollador 3 (Thiago Gomez Kehler - UI de Controles, Integración Drag & Drop & Tarjetas de Métricas)**
- **Flujo Drag & Drop en `Laboratorio.tsx` (Subfase 1.3):** Conectar la zona interactiva de carga de imágenes de marcas con la petición asíncrona al backend FastAPI.
- **Panel de Ajustes Top-K & Diales (Subfase 1.4):** Conectar los controles en `Laboratorio.tsx` (Slider de Top-K y umbrales) para actualizar dinámicamente la petición de inferencia.
- **Grilla de Resultados & Tarjetas de Similitud (Subfase 1.4):** Renderizar las tarjetas visuales del Top-K con badges de porcentaje de coincidencia, distancia métrica relacional y comparativa visual.

---

## 3. Plan de Comprobación, Validación y Testing

Las validaciones se ejecutan de manera progresiva por cada subfase y se concluyen con la prueba E2E de la Subfase 1.5.

---

## 4. Protocolo de Subida a Git
Al culminar la Subfase 1.5 y comprobar que la compilación `npm run build` y la inferencia funcionan al 100%, **se consulta al usuario para autorizar la subida (push) a Git**.
