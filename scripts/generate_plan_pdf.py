import os
import subprocess
import sys

def compile_html_to_pdf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_input_path = os.path.join(script_dir, "Contexto", "Plan_Implementacion_GIBD_2026.html")
    pdf_output_path = os.path.join(script_dir, "Contexto", "Plan_Implementacion_GIBD_2026.pdf")
    
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    
    print(f"Buscando HTML en: {html_input_path}")
    print(f"Destino del PDF en: {pdf_output_path}")
    
    if not os.path.exists(html_input_path):
        print(f"Error: El archivo HTML de entrada no existe en {html_input_path}")
        sys.exit(1)
        
    if not os.path.exists(chrome_path):
        print(f"Error: Google Chrome no se encontró en {chrome_path}")
        sys.exit(1)
        
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_output_path}",
        f"file:///{html_input_path.replace(os.sep, '/')}"
    ]
    
    print("Ejecutando Google Chrome Headless para compilar el PDF...")
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(pdf_output_path) and os.path.getsize(pdf_output_path) > 0:
            print("==========================================================================")
            print(" ¡ÉXITO! PDF del Plan de Implementación generado perfectamente.")
            print(f" Ubicación: {pdf_output_path}")
            print(f" Tamaño: {os.path.getsize(pdf_output_path)} bytes")
            print("==========================================================================")
        else:
            print("Error: El comando se ejecutó pero el archivo PDF de salida está vacío o no se creó.")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print("Error durante la compilación con Chrome Headless:")
        print(e.stderr.decode("utf-8", errors="ignore"))
        sys.exit(1)

if __name__ == "__main__":
    compile_html_to_pdf()
