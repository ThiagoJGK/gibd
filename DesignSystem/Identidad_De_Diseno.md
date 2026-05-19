# Identidad de Diseño y UI/UX - GIBD WEB

Este documento establece las normativas visuales, estéticas y técnicas para la interfaz de la aplicación web y móvil del Grupo de Investigación en Big Data (GIBD). Todo nuevo desarrollo o componente debe apegarse estrictamente a estas directrices.

## 1. Paleta de Colores (Tema Oscuro por Defecto)

El diseño se apoya en un contraste fuerte entre fondos profundos, verdes orgánicos y acentos de alta energía.

*   **Negro Base (Fondo Principal):** `#0A0A0A` (Un negro casi puro, ideal para pantallas OLED y contraste infinito).
*   **Morado Profundo (Color Secundario / Fondos de tarjetas / Superficies):** `#1A1124` a `#261633` (Morado profundo, orgánico pero elegante. Sirve para diferenciar secciones del fondo negro sin ser disruptivo).
*   **Naranja Vibrante (Acento / Call to Action / Decoraciones):** `#FF5500` (Naranja puro, eléctrico y sumamente llamativo. Usado en botones principales, decoraciones, gradientes de fondo y elementos clave).
*   **Neutros de Soporte (Texto y Bordes):**
    *   Texto Principal: `#F2F2F2` (Blanco suave para no cansar la vista).
    *   Texto Secundario: `#A3A3A3` (Gris medio para jerarquía menor).
    *   Bordes "Flat" / Orgánicos: `#3A234A` a `#333333` (Para delimitar contenedores sin usar sombras).

## 2. Tipografía (Escala y Emparejamiento)

Se utilizará una tipografía **sans-serif geométrica, moderna y altamente legible**. Opciones recomendadas: **Outfit**, **Plus Jakarta Sans** o **Poppins**.

*   **Jerarquía de Títulos (Headings):**
    *   Estilo: Gigantes, pesados (Bold / ExtraBold / Black).
    *   Web/Desktop: `h1` a `5rem` (80px), `h2` a `3.5rem` (56px).
    *   Mobile: Escalar proporcionalmente. `h1` a `3rem` (48px) para no desbordar la pantalla, manteniendo el peso *Black*.
*   **Cuerpo de Texto (Body):**
    *   Estilo: Regular o Medium. Priorizar la legibilidad con alto contraste.
    *   Tamaño base: `1rem` (16px) a `1.125rem` (18px) para web, `1rem` para móvil.
    *   Interlineado (Line-height): `1.5` a `1.6` para dejar "respirar" al texto (espacio en blanco).

## 3. Geometría y Estética "Pill" (Cápsula)

El enfoque principal de la UI es la geometría redondeada en su extremo máximo, aplicada en un entorno completamente plano (*Flat Design*).

*   **Border-Radius Universal:** Todos los contenedores, botones, tarjetas del laboratorio, inputs de texto y barras de navegación deben usar `border-radius: 9999px` (o un valor `px` lo suficientemente grande para formar una píldora perfecta en los extremos, o bordes completamente redondeados en contenedores grandes).
*   **Estilo Flat (Plano):**
    *   **NO** se utilizarán sombras (`box-shadow`) para dar profundidad.
    *   La separación de los elementos se logrará mediante el contraste entre el *Negro Base* y el *Verde Bosque*, o utilizando bordes sólidos y finos (`border: 1px solid #333`).
*   **Paddings (Espaciado Interno):** Al usar esquinas tan redondeadas, los `padding-left` y `padding-right` de los botones y contenedores deben ser amplios (mínimo el doble del padding vertical, ej: `padding: 12px 32px;`) para evitar que el contenido "choque" con la curva visual.

## 4. Comportamiento y Animaciones (Microinteracciones)

La interacción debe sentirse fluida y viva, respondiendo a las acciones del usuario.

*   **Efecto "Onda" (Ripple Effect):** Al hacer clic o tap en un elemento interactivo (como un botón Naranja), se debe desplegar una sutil animación de expansión radial (como una gota cayendo en el agua) en el interior del botón.
*   **Transiciones de Estado (Hover/Focus):** Los cambios de color de fondo o de borde deben usar animaciones fluidas de corta duración. `transition: background-color 0.2s ease, transform 0.2s ease;`.
*   **Fluidez Optimizada:** Para entradas a nuevas páginas o aparición de elementos en el DOM (como resultados en el Laboratorio), se debe usar un suave desvanecimiento hacia arriba (`fade-in-up`).
*   **Regla de Oro en Animación:** **NUNCA** animar propiedades que causen *Reflow* o repintado pesado en el navegador (como `width`, `height`, `margin`, `top/left`). Animaremos **únicamente** `transform` (translaciones, escalas) y `opacity`.

## 5. Diseño Responsivo (Web vs Móvil)

El diseño es inherentemente "Mobile-First" y fluido. La misma línea estética rige para ambas resoluciones.

*   **Móvil (Mobile):** 
    *   Uso agresivo del espacio vertical.
    *   El formato "Pill" es ideal para botones que abarcan todo el ancho en móvil.
    *   Las tarjetas grandes (que en web tienen esquinas super redondeadas) en móvil reducirán levemente su curva (`border-radius: 32px`) si empiezan a "robar" demasiado espacio útil de la pantalla pequeña.
    *   El menú de navegación web pasará a ser una barra de navegación inferior flotante tipo cápsula, o un menú de hamburguesa fullscreen, para máxima ergonomía con el pulgar.
*   **Escritorio (Desktop):** 
    *   Aprovechamiento de layouts en grilla, manteniendo un límite de ancho máximo (ej. `max-width: 1280px`) para evitar que el texto se vuelva ilegible en monitores ultra anchos.
    *   El Laboratorio aprovechará el espacio horizontal para presentar controles a la izquierda y gráficos interactivos a la derecha.

## 6. Rendimiento y Optimización de Renderizado

Dado el enfoque en animaciones fluidas y colores sólidos de alto contraste:
*   **Hardware Acceleration:** Forzar el renderizado por GPU en los elementos animados de la Landing Page usando `transform: translateZ(0)` o la propiedad `will-change`.
*   **Carga de Fuentes:** Utilizar la directiva `font-display: swap` en las importaciones CSS para asegurar que el contenido sea visible inmediatamente sin retrasos perceptibles en la primera carga.
*   **Evitar DOMs densos:** Para las pantallas del Laboratorio que requieran renderizar muchos nodos (ej. gráficos de reducción dimensional), evaluar el uso de `Canvas` o `WebGL` en lugar de miles de divs en HTML para mantener los 60 FPS (o 120 FPS en pantallas modernas).
