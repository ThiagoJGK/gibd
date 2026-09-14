import hashlib

# Monkeypatch hashlib.md5 to ignore 'usedforsecurity' keyword argument
original_md5 = hashlib.md5
def patched_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return original_md5(*args, **kwargs)
hashlib.md5 = patched_md5

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor

class RoundedCard(Flowable):
    """
    A custom ReportLab Flowable that draws a beautifully rounded container (Pill aesthetic)
    and renders a list of flowables inside it. Clamps corner radius to prevent visual distortions.
    """
    def __init__(self, content_flowables, width, bg_color, border_color, border_width=1.5, corner_radius=16, padding=14):
        Flowable.__init__(self)
        self.contents = content_flowables
        self.width = width
        self.bg_color = bg_color
        self.border_color = border_color
        self.border_width = border_width
        self.corner_radius = corner_radius
        self.padding = padding
        
        # Calculate total height needed based on wrapping contents
        self.height = self._calculate_height()
        
        # Prevent ReportLab distortion by clamping corner_radius to at most half of the smallest dimension
        max_radius = min(self.width, self.height) / 2.0
        if self.corner_radius > max_radius:
            self.corner_radius = max_radius

    def _calculate_height(self):
        h = self.padding * 2
        for f in self.contents:
            w, fh = f.wrap(self.width - (self.padding * 2), 1000)
            h += fh + 5  # added spacing between elements
        return h

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        
        # Draw background rounded rectangle (Pill-inspired modern geometry)
        canvas.setFillColor(self.bg_color)
        canvas.setStrokeColor(self.border_color)
        canvas.setLineWidth(self.border_width)
        canvas.roundRect(0, 0, self.width, self.height, self.corner_radius, fill=1, stroke=1)
        
        # Draw contents from top to bottom
        y = self.height - self.padding
        for f in self.contents:
            w, fh = f.wrap(self.width - (self.padding * 2), 1000)
            y -= fh
            f.drawOn(canvas, self.padding, y)
            y -= 5  # space between items
            
        canvas.restoreState()

def draw_cover_background(canvas, doc):
    canvas.saveState()
    # Dark background matching GIBD's #0A0A0A web aesthetic
    canvas.setFillColor(HexColor('#0A0A0A'))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
    
    # Glowing decorative gradient effect (using concentric overlapping transparent circles)
    # Top-left glowing aura
    for r in range(450, 150, -35):
        canvas.setFillColor(HexColor('#1A082E'))
        canvas.setFillAlpha(0.06)
        canvas.circle(0, doc.pagesize[1], r, fill=1, stroke=0)
        
    # Bottom-right glowing aura
    for r in range(350, 100, -30):
        canvas.setFillColor(HexColor('#230E3D'))
        canvas.setFillAlpha(0.06)
        canvas.circle(doc.pagesize[0], 0, r, fill=1, stroke=0)
        
    # Restore opacity for subsequent elements
    canvas.setFillAlpha(1.0)
    
    # Elegant orange glowing neon line at the top border
    canvas.setStrokeColor(HexColor('#FF5500'))
    canvas.setStrokeAlpha(0.15)
    canvas.setLineWidth(8)
    canvas.line(0, doc.pagesize[1] - 8, doc.pagesize[0], doc.pagesize[1] - 8)
    
    canvas.setStrokeAlpha(1.0)
    canvas.setLineWidth(2)
    canvas.line(0, doc.pagesize[1] - 8, doc.pagesize[0], doc.pagesize[1] - 8)
    
    # Subtle flat geometric separator line above metadata
    canvas.setStrokeColor(HexColor('#3B1B5F'))
    canvas.setStrokeAlpha(0.4)
    canvas.setLineWidth(1)
    canvas.line(100, doc.pagesize[1]/2 - 130, doc.pagesize[0] - 100, doc.pagesize[1]/2 - 130)
    
    canvas.restoreState()

def draw_page_decorations(canvas, doc):
    canvas.saveState()
    # Dark web background
    canvas.setFillColor(HexColor('#0A0A0A'))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
    
    # Left vertical purple accent line
    canvas.setStrokeColor(HexColor('#2E134D'))
    canvas.setLineWidth(2)
    canvas.line(40, 50, 40, doc.pagesize[1] - 50)
    
    # Top-right glowing orange pill (header detail)
    canvas.setFillColor(HexColor('#FF5500'))
    canvas.roundRect(doc.pagesize[0] - 70, doc.pagesize[1] - 40, 30, 8, 4, fill=1, stroke=0)
    
    # Bottom footer information
    canvas.setFont('Helvetica-Bold', 8)
    canvas.setFillColor(HexColor('#A78BFA'))
    canvas.drawString(55, 30, "GIBD 2026  |  MANIFIESTO DE LA PLATAFORMA")
    
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(HexColor('#8B5CF6'))
    canvas.drawRightString(doc.pagesize[0] - 40, 30, f"PÁGINA {doc.page}")
    canvas.restoreState()

def create_manifest_pdf(filename="Contexto/Manifiesto_GIBD_2026.pdf"):
    # Margins and Document setup (Aesthetic flat layouts)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Color palette tokens matching the web identity
    c_white = HexColor('#FFFFFF')
    c_lavender = HexColor('#F3E8FF')
    c_purple_light = HexColor('#A78BFA')
    c_purple_med = HexColor('#8B5CF6')
    c_orange = HexColor('#FF5500')
    c_card_bg = HexColor('#140A24')      # Morado de superficie
    c_card_border = HexColor('#2F154F')  # Línea fina para superficies
    
    # Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=38,
        leading=46,
        textColor=c_white,
        alignment=1,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=c_orange,
        alignment=1,
        spaceAfter=35
    )
    
    metadata_style = ParagraphStyle(
        'CoverMetadata',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=16,
        textColor=c_purple_light,
        alignment=1
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=19,
        leading=25,
        textColor=c_orange,
        spaceBefore=22,
        spaceAfter=12,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=18,
        textColor=c_purple_light,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14.5,
        textColor=c_lavender,
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=c_lavender,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    card_title_style = ParagraphStyle(
        'CardTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_white
    )
    
    card_body_style = ParagraphStyle(
        'CardBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=c_lavender
    )
    
    story = []
    
    # ------------------ COVER PAGE ------------------
    story.append(Spacer(1, 120))
    story.append(Paragraph("PLATAFORMA INSTITUCIONAL Y DE EXPERIMENTACIÓN", subtitle_style))
    story.append(Paragraph("GIBD 2026", title_style))
    story.append(Spacer(1, 15))
    
    # Extremely rounded "Pill" badge for the cover page title
    # Clamping is handled automatically inside RoundedCard to make it a perfect pill shape!
    cover_badge_p = Paragraph("<b>MANIFIESTO DE DESARROLLO Y DISEÑO</b>", ParagraphStyle('BadgeTxt', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=12, textColor=c_white, alignment=1))
    cover_badge_card = RoundedCard([cover_badge_p], width=260, bg_color=c_orange, border_color=c_orange, corner_radius=99, padding=10)
    
    # Center badge horizontally using a table layout
    badge_table = Table([[cover_badge_card]], colWidths=[500], hAlign='CENTER')
    badge_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(badge_table)
    
    story.append(Spacer(1, 180))
    
    metadata_text = """
    <b>Grupo de Investigación en Big Data (GIBD)</b><br/>
    Universidad Tecnológica Nacional (UTN FRCU)<br/>
    <font color='#A78BFA'>React · Vite · Supabase · Python FastAPI · Gemini AI</font><br/>
    <font color='#8B5CF6'>Versión 2.0 · Mayo 2026</font>
    """
    story.append(Paragraph(metadata_text, metadata_style))
    story.append(PageBreak())
    
    # ------------------ PAGE 2: OBJECTIVES & VISION ------------------
    story.append(Paragraph("1. Visión General del Proyecto", h1_style))
    story.append(Paragraph(
        "La <b>Plataforma Institucional y de Experimentación del GIBD</b> (Grupo de Investigación en Big Data de la UTN FRCU) "
        "es una solución integral de software diseñada para democratizar el acceso a las investigaciones del grupo, visibilizar "
        "sus productos tecnológicos y proporcionar un entorno interactivo (<b>\"Laboratorio\"</b>) donde los usuarios puedan poner a prueba "
        "los modelos de Inteligencia Artificial desarrollados por los investigadores.",
        body_style
    ))
    story.append(Paragraph(
        "El proyecto está concebido bajo una estricta <b>arquitectura \"Zero-Cost\"</b> inicial, aprovechando servicios en la nube "
        "de nivel gratuito (Vercel, Supabase, Render/Hugging Face) para minimizar la carga presupuestaria de la universidad, pero "
        "manteniendo estándares de ingeniería de software corporativos de primer nivel.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("2. Objetivos Principales", h1_style))
    
    objectives = [
        ("Visibilidad Académica", "Centralizar en un solo portal todas las noticias, papers y avances científicos del GIBD de forma atractiva y premium."),
        ("Transferencia Tecnológica", "Permitir que usuarios no técnicos interactúen en tiempo real con modelos complejos (ej. Redes Siamesas para marcas de ganado) sin requerir instalación local."),
        ("Optimización Operativa", "Proveer un CMS de autogestión para investigadores, permitiendo publicar noticias, subir papers y definir colaboradores de forma simple."),
        ("Diseño de Vanguardia", "Romper con el molde de las 'webs académicas aburridas' mediante una UI/UX inmersiva, oscura, geométrica y de alto rendimiento.")
    ]
    
    for title, desc in objectives:
        bullet_html = f"<b>&bull; {title}:</b> {desc}"
        story.append(Paragraph(bullet_html, bullet_style))
        
    story.append(PageBreak())
    
    # ------------------ PAGE 3: MODULES A & B ------------------
    story.append(Paragraph("3. Módulos y Funcionalidades del Sistema", h1_style))
    
    # Card 1: Landing Page
    landing_text = [
        "El punto de entrada público, diseñado para alto impacto visual y conversión de lectura.",
        "• <b>Hero Section:</b> Presentación moderna del grupo con CTA directo al Laboratorio interactivo.",
        "• <b>Grupos de Investigación:</b> Exposición visual en el inicio de los distintos grupos internos de trabajo (ej. encargados de video, sonido, etc.).",
        "• <b>Noticias Dinámicas:</b> Grilla interactiva de novedades extraídas en tiempo real desde la base de datos.",
        "• <b>Acerca de / Misión:</b> Resumen dinámico de las áreas de estudio (Deep Learning, Análisis Numérico, etc.)."
    ]
    
    def make_rounded_card(card_title, content_list, badge="PÚBLICO"):
        title_html = f"<b>{card_title.upper()}</b>"
        if badge:
            badge_color = '#FF5500' if badge in ['ADMIN', 'NUEVO'] else '#A78BFA'
            title_html += f"   <font size='8' color='{badge_color}'><b> [ {badge} ]</b></font>"
            
        story_content = [Paragraph(title_html, card_title_style), Spacer(1, 6)]
        for item in content_list:
            story_content.append(Paragraph(item, card_body_style))
            
        # Return our custom RoundedCard Flowable (corner_radius=20 makes it look heavily rounded/pill style!)
        return RoundedCard(story_content, width=490, bg_color=c_card_bg, border_color=c_card_border, corner_radius=20, padding=15)

    story.append(Paragraph("A. Landing Page Institucional", h2_style))
    story.append(make_rounded_card("Landing Page", landing_text, "PÚBLICO"))
    
    story.append(Spacer(1, 15))
    
    # Card 2: Laboratorio
    lab_text = [
        "El núcleo interactivo de la plataforma. Una consola técnica para ejecutar inferencias sobre modelos de IA.",
        "• <b>Marcas de Ganado:</b> Carga interactiva (Drag & Drop) de marcas de ganado vacuno para identificar similitudes utilizando Redes Neuronales Siamesas (One-Shot Learning) y ver distancias relativas.",
        "• <b>PTAH-Jurídico:</b> Motor de búsqueda semántica (NLP) para consultas sobre normativas jurídicas.",
        "• <b>OBC (Huesos de Oráculo):</b> Reconocimiento óptico de glifos y caracteres antiguos sobre huesos.",
        "• <b>Sonido / Video:</b> Módulos preparados para futuras integraciones multimodales del grupo.",
        "• <b>Panel Lateral Técnico:</b> Ajuste unificado de hiperparámetros (Top-K, umbrales de similitud)."
    ]
    story.append(Paragraph("B. Laboratorio de Experimentación", h2_style))
    story.append(make_rounded_card("Laboratorio IA", lab_text, "INTERACTIVO"))
    
    story.append(PageBreak())
    
    # ------------------ PAGE 4: MODULES C & D ------------------
    story.append(Paragraph("3. Módulos y Funcionalidades (Cont.)", h1_style))
    
    # Card 3: Repositorio
    repo_text = [
        "Biblioteca digital para democratizar el conocimiento científico generado por el GIBD.",
        "• <b>Listado de Investigaciones:</b> Grilla interactiva de papers con títulos, abstracts y filtros temáticos.",
        "• <b>Colaboradores:</b> Mapeo visual de los autores y colaboradores involucrados en cada investigación, conectados directamente con la base de datos de alumnos y docentes.",
        "• <b>Descargas Directas:</b> Acceso rápido a archivos PDF enlazados a Supabase Storage (ej. papers CACIC 2023)."
    ]
    story.append(Paragraph("C. Repositorio Académico (Papers)", h2_style))
    story.append(make_rounded_card("Biblioteca Científica", repo_text, "PÚBLICO"))
    
    story.append(Spacer(1, 15))
    
    # Card 4: Panel de Administración
    admin_text = [
        "Panel exclusivo para la gestión ágil del contenido del sitio, protegido bajo roles seguros.",
        "• <b>Control de Acceso y Permisos:</b> Restricción de edición únicamente a usuarios seleccionados como administradores (asociados a un email). La lista de correos autorizados es totalmente configurable.",
        "• <b>Gestión de Contenido (CRUD):</b> Creación, edición y borrado de noticias y papers. Permite asociar dinámicamente colaboradores a cada investigación desde la base de alumnos.",
        "• <b>Gestión de Grupos (CRUD):</b> ABM completo de grupos internos de trabajo (ej. grupo video, sonido, etc.).",
        "• <b>Equipo y Roles (CRUD):</b> Registro de docentes y alumnos. Para alumnos, se captura: foto, nombre, email y perfil de LinkedIn.",
        "• <b>Asistente de Papers (Gemini AI):</b> Extracción de metadatos, generación de abstracts y prompts de ilustración."
    ]
    story.append(Paragraph("D. Panel de Administración", h2_style))
    story.append(make_rounded_card("Consola de Administración", admin_text, "ADMIN"))
    
    story.append(PageBreak())
    
    # ------------------ PAGE 5: ARCHITECTURE & DESIGN SYSTEM ------------------
    story.append(Paragraph("4. Arquitectura Tecnológica", h1_style))
    
    arch_items = [
        ("1. Frontend (Capa de Presentación y UI)", "React.js + Vite + TypeScript. Interfaz Mobile-First, animaciones ultraligeras con Vanilla CSS y gestión de subida de archivos (react-dropzone). Hosting optimizado en Vercel."),
        ("2. Backend BaaS (Datos y Autenticación)", "Supabase (PostgreSQL + Auth + Storage). Gestión ágil de bases de datos relacionales, autenticación robusta mediante JWT con RLS (Row Level Security) y almacenamiento de archivos."),
        ("3. Backend de IA (AI API Gateway)", "FastAPI + PyTorch/TensorFlow en Python. Inferencia en tiempo real sobre redes neuronales complejas y entrega de datos estandarizados vía JSON. Alojamiento flexible en Render o servidores locales.")
    ]
    
    for title, desc in arch_items:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        story.append(Paragraph(desc, body_style))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("5. Identidad Visual & UX (Design System)", h1_style))
    story.append(Paragraph(
        "La estética visual del GIBD es un componente fundamental e identitario de la plataforma, "
        "combinando dinamismo, minimalismo y un alto contraste de colores vibrantes sobre superficies oscuras.",
        body_style
    ))
    
    story.append(Spacer(1, 5))
    
    # 3 Highly rounded color swatches side by side!
    swatch_title_style = ParagraphStyle('SwatchTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=c_white, alignment=1)
    swatch_body_style = ParagraphStyle('SwatchBody', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10, textColor=c_lavender, alignment=1)
    
    swatch_1 = RoundedCard([
        Paragraph("<b>FONDO</b>", swatch_title_style),
        Spacer(1, 4),
        Paragraph("<font color='#FFFFFF'>NEGRO PURO</font><br/><b>#0A0A0A</b>", swatch_body_style)
    ], width=145, bg_color=HexColor('#0A0A0A'), border_color=c_card_border, corner_radius=20, padding=12)
    
    swatch_2 = RoundedCard([
        Paragraph("<b>SUPERFICIES</b>", swatch_title_style),
        Spacer(1, 4),
        Paragraph("<font color='#F3E8FF'>MORADO ELEGANTE</font><br/><b>#180A24</b>", swatch_body_style)
    ], width=145, bg_color=HexColor('#180A24'), border_color=c_card_border, corner_radius=20, padding=12)
    
    swatch_3 = RoundedCard([
        Paragraph("<b>ACENTOS Y CTAs</b>", swatch_title_style),
        Spacer(1, 4),
        Paragraph("<font color='#FFFFFF'>NARANJA VIBRANTE</font><br/><b>#FF5500</b>", swatch_body_style)
    ], width=145, bg_color=HexColor('#FF5500'), border_color=HexColor('#FF5500'), corner_radius=20, padding=12)
    
    # Layout swatches horizontally using a Table
    swatches_table = Table([[swatch_1, swatch_2, swatch_3]], colWidths=[160, 160, 160], hAlign='CENTER')
    swatches_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    
    story.append(swatches_table)
    story.append(Spacer(1, 15))
    
    ui_rules = [
        "<b>Geometría 'Pill' (Cápsula):</b> Redondeo extremo (<code>border-radius: 9999px</code>) en botones, etiquetas, selectores y contenedores interactivos para un look sumamente moderno.",
        "<b>Estética Flat y Contraste:</b> Eliminación absoluta de sombras proyectadas difusas (<code>box-shadow</code>). Las delimitaciones y jerarquías se crean mediante bordes finos y el fuerte contraste entre el fondo oscuro y las superficies moradas.",
        "<b>Micro-Animaciones Fluidas:</b> Transiciones CSS suaves para estados interactivos (<code>hover</code>, <code>focus</code>) y animaciones de entrada controladas (<code>fade-in-up</code>) para optimizar el rendimiento de renderizado en navegadores."
    ]
    
    for rule in ui_rules:
        story.append(Paragraph(f"<b>&bull;</b> {rule}", bullet_style))
        
    # Build Document
    doc.build(
        story,
        onFirstPage=draw_cover_background,
        onLaterPages=draw_page_decorations
    )

if __name__ == "__main__":
    create_manifest_pdf()
    print("PDF creado con éxito con estética redondeada.")
