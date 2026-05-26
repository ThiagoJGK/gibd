import os
import subprocess
import sys

def compile_html_to_pdf():
    # Define absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_input_path = os.path.join(script_dir, "Contexto", "workflow_guide.html")
    pdf_output_path = os.path.join(script_dir, "Contexto", "Guia_Trabajo_Remoto_GIBD.pdf")
    
    # Chrome path
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    
    print(f"Buscando HTML en: {html_input_path}")
    print(f"Destino del PDF en: {pdf_output_path}")
    
    if not os.path.exists(html_input_path):
        print(f"Error: El archivo HTML de entrada no existe en {html_input_path}")
        sys.exit(1)
        
    if not os.path.exists(chrome_path):
        print(f"Error: Google Chrome no se encontró en {chrome_path}")
        sys.exit(1)
        
    # Build Chrome Headless CLI command
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",  # Hides default header/footer text completely
        f"--print-to-pdf={pdf_output_path}",
        html_input_path
    ]
    
    print("Ejecutando Google Chrome Headless para compilar el PDF con pixel-perfect precision...")
    try:
        # Run process synchronously
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Verify output
        if os.path.exists(pdf_output_path) and os.path.getsize(pdf_output_path) > 0:
            print("==========================================================================")
            print(" ¡ÉXITO! PDF de la Guía de Trabajo Remoto generado perfectamente.")
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
