from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


HTML = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Buscador de patentes GIBD</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #101010; color: #f2f2f2; }
    main { width: min(920px, calc(100% - 32px)); margin: 24px auto; padding: 28px; border: 1px solid #333; border-radius: 24px; background: #1b1b1b; }
    .eyebrow { color: #ff5500; font-size: .75rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    h1 { margin: 8px 0; font-size: clamp(1.7rem, 4vw, 2.6rem); }
    p { color: #aaa; line-height: 1.55; }
    .workspace { display: grid; grid-template-columns: minmax(220px, .8fr) minmax(280px, 1.2fr); gap: 20px; margin-top: 24px; }
    .panel { padding: 20px; border: 1px solid #333; border-radius: 16px; background: #141414; }
    .dropzone { display: grid; min-height: 220px; place-items: center; border: 1px dashed #666; border-radius: 12px; text-align: center; cursor: pointer; overflow: hidden; }
    .dropzone:hover { border-color: #ff5500; }
    .dropzone input { display: none; }
    .preview { display: none; width: 100%; height: 220px; object-fit: contain; background: #090909; }
    .upload-shell { position: relative; }
    .clear-image { display: none; position: absolute; z-index: 2; top: 10px; right: 10px; width: 34px; height: 34px; margin: 0; padding: 0; align-items: center; justify-content: center; border: 1px solid #fff4; background: #111e; font-family: Arial, sans-serif; font-size: 1.25rem; font-weight: 400; line-height: 1; }
    .clear-image:hover { background: #ff5500; }
    .icon { font-size: 2.4rem; color: #ff5500; }
    label { display: block; margin-top: 20px; font-weight: 700; }
    input[type=range] { width: 100%; margin-top: 12px; accent-color: #ff5500; }
    .range-value { color: #ff5500; float: right; }
    button { width: 100%; margin-top: 24px; padding: 13px 18px; border: 0; border-radius: 999px; background: #ff5500; color: white; font: inherit; font-weight: 800; cursor: pointer; }
    button:hover { background: #ff731f; }
    .status { display: flex; align-items: center; gap: 8px; margin-top: 16px; color: #aaa; font-size: .9rem; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #35c978; }
    .results { display: grid; gap: 10px; margin-top: 18px; }
    .result { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px; border-left: 3px solid #ff5500; border-radius: 8px; background: #222; }
    .result-info { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .result-preview { position: relative; display: block; width: 76px; height: 58px; flex: 0 0 auto; margin: 0; padding: 0; border: 0; border-radius: 6px; background: #090909; cursor: zoom-in; overflow: hidden; }
    .result-thumb { display: block; width: 100%; height: 100%; border-radius: 6px; object-fit: cover; }
    .view-mark { position: absolute; inset: 0; display: grid; place-items: center; background: #0008; opacity: 0; transition: opacity 160ms ease; pointer-events: none; }
    .result-preview:hover .view-mark, .result-preview:focus-visible .view-mark { opacity: 1; }
    .view-icon { width: 24px; height: 24px; color: #fff; filter: drop-shadow(0 1px 2px #000); }
    .result strong { display: block; }
    .result small { color: #999; }
    .score { color: #ff5500; font-weight: 800; white-space: nowrap; }
    .modal { display: none; position: fixed; z-index: 10; inset: 0; place-items: center; padding: 20px; background: #000c; }
    .modal.open { display: grid; }
    .modal-content { position: relative; width: min(560px, 100%); padding: 20px; border: 1px solid #444; border-radius: 16px; background: #1b1b1b; text-align: center; }
    .modal-image { width: 100%; max-height: 55vh; object-fit: contain; border-radius: 10px; background: #090909; }
    .modal-close { display: flex; position: absolute; top: 10px; right: 10px; width: 36px; height: 36px; margin: 0; padding: 0; align-items: center; justify-content: center; border: 1px solid #fff5; background: #111e; font-family: Arial, sans-serif; font-size: 1.3rem; font-weight: 400; line-height: 1; }
    .modal-close:hover { background: #ff5500; }
    .modal-title { margin: 14px 0 0; color: #ff5500; font-size: 1.2rem; font-weight: 800; }
    @media (max-width: 680px) { main { padding: 20px; } .workspace { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">GIBD / Laboratorio multimodal</div>
    <h1>Buscador visual de patentes</h1>
    <p>Interfaz de demostración servida por Python para consultar un índice visual de patentes de automóviles.</p>
    <div class="workspace">
      <section class="panel">
        <div class="upload-shell">
          <label class="dropzone" for="image">
            <span id="upload-message"><span class="icon">＋</span><br><strong id="file-label">Seleccionar una imagen</strong><br><small>JPG, PNG o WEBP</small></span>
            <img id="preview" class="preview" alt="Miniatura de la patente cargada">
            <input id="image" type="file" accept="image/*">
          </label>
          <button id="clear-image" class="clear-image" type="button" aria-label="Quitar imagen" title="Quitar imagen">×</button>
        </div>
        <label for="topK">Cantidad de resultados <span class="range-value" id="value">5</span></label>
        <input id="topK" type="range" min="1" max="10" value="5">
        <button id="analyze" type="button">Analizar imagen</button>
        <div class="status"><span class="dot"></span><span id="status">Motor listo para recibir una consulta</span></div>
      </section>
      <section class="panel">
        <div class="eyebrow">Patentes similares</div>
        <div id="results" class="results"><p>Los resultados similares aparecerán aquí después del análisis.</p></div>
      </section>
    </div>
    <div id="image-modal" class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div class="modal-content">
        <button id="modal-close" class="modal-close" type="button" aria-label="Cerrar imagen ampliada" title="Cerrar">×</button>
        <img id="modal-image" class="modal-image" alt="Imagen ampliada de patente">
        <div id="modal-title" class="modal-title"></div>
      </div>
    </div>
  </main>
  <script>
    const slider = document.querySelector('#topK');
    const value = document.querySelector('#value');
    const status = document.querySelector('#status');
    const image = document.querySelector('#image');
    const fileLabel = document.querySelector('#file-label');
    const preview = document.querySelector('#preview');
    const uploadMessage = document.querySelector('#upload-message');
    const clearImage = document.querySelector('#clear-image');
    const results = document.querySelector('#results');
    const analyze = document.querySelector('#analyze');
    const imageModal = document.querySelector('#image-modal');
    const modalImage = document.querySelector('#modal-image');
    const modalTitle = document.querySelector('#modal-title');
    const modalClose = document.querySelector('#modal-close');
    const samples = ['AB 482 CD', 'AC 719 EF', 'AD 305 GH', 'AE 861 JK', 'AF 194 LM', 'AG 572 NP', 'AH 638 QR', 'AJ 207 ST', 'AK 943 UV', 'AL 516 WX'];
    const thumbnailColors = ['#d94f28', '#2878b8', '#4b9b65', '#b68a35', '#824f9f', '#c24f6c', '#397b82', '#8c633e', '#5c6fc0', '#7d8b3d'];
    const makeThumbnail = (plate, index) => `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 152 116"><rect width="152" height="116" fill="#111"/><rect x="9" y="18" width="134" height="80" rx="10" fill="${thumbnailColors[index]}"/><rect x="18" y="34" width="116" height="48" rx="4" fill="#f3f0dd"/><text x="76" y="65" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#161616">${plate}</text><circle cx="27" cy="91" r="3" fill="#ff5500"/><circle cx="125" cy="91" r="3" fill="#ff5500"/></svg>`)}`;
    slider.addEventListener('input', () => {
      value.textContent = slider.value;
    });
    image.addEventListener('change', () => {
      const file = image.files[0];
      if (!file) {
        fileLabel.textContent = 'Seleccionar una imagen';
        preview.style.display = 'none';
        uploadMessage.style.display = 'block';
        clearImage.style.display = 'none';
        return;
      }
      fileLabel.textContent = 'Imagen seleccionada';
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
      uploadMessage.style.display = 'none';
      clearImage.style.display = 'flex';
    });
    clearImage.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      image.value = '';
      fileLabel.textContent = 'Seleccionar una imagen';
      preview.removeAttribute('src');
      preview.style.display = 'none';
      uploadMessage.style.display = 'block';
      clearImage.style.display = 'none';
      status.textContent = 'Motor listo para recibir una consulta';
    });
    const closeModal = () => {
      imageModal.classList.remove('open');
      modalImage.removeAttribute('src');
    };
    modalClose.addEventListener('click', closeModal);
    imageModal.addEventListener('click', (event) => {
      if (event.target === imageModal) closeModal();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closeModal();
    });
    results.addEventListener('click', (event) => {
      const previewButton = event.target.closest('.result-preview');
      if (!previewButton) return;
      const thumbnail = previewButton.querySelector('.result-thumb');
      modalImage.src = thumbnail.src;
      modalImage.alt = thumbnail.alt;
      modalTitle.textContent = thumbnail.alt.replace('Miniatura patente ', 'Patente ');
      imageModal.classList.add('open');
    });
    analyze.addEventListener('click', () => {
      analyze.disabled = true;
      analyze.textContent = 'Procesando...';
      status.textContent = 'Extrayendo características de la patente...';
      results.innerHTML = '<p>Consultando índice vectorial...</p>';
      setTimeout(() => {
        const count = Number(slider.value);
        results.innerHTML = samples.slice(0, count).map((sample, index) => `<div class="result"><span class="result-info"><button class="result-preview" type="button" aria-label="Ampliar patente ${sample}"><img class="result-thumb" src="${makeThumbnail(sample, index)}" alt="Miniatura patente ${sample}"><span class="view-mark" aria-hidden="true"><svg class="view-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"></circle><path d="m20 20-4-4"></path></svg></span></button><span><strong>${sample}</strong><small>Patente similar / vehículo ${2024 - index}</small></span></span><span class="score">${98 - index * 3}%</span></div>`).join('');
        status.textContent = `Análisis completado: ${count} coincidencias encontradas`;
        analyze.disabled = false;
        analyze.textContent = 'Analizar imagen';
      }, 900);
    });
  </script>
</body>
</html>"""


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "healthy", "service": "GIBD similarity demo"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path != "/" and self.path != "/index.html":
            self.send_error(404)
            return

        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8010), DemoHandler)
    print("Demo Python GIBD disponible en http://localhost:8010")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
