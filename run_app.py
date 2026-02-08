import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    
    print("Starting Fake News Detection System...")
    
    # Iniciar Backend API
    print("Starting Backend API (FastAPI)...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(project_root),
        env=os.environ.copy()
    )
    
    #  Esperar a que la API se inicialice (solo para propósitos de ejemplo)
    print("Waiting for API to initialize...")
    time.sleep(10) 
    
    #  Iniciar Frontend UI
    print("Starting Frontend UI (Gradio)...")
    ui_process = subprocess.Popen(
        [sys.executable, "src/ui/gradio_app.py"],
        cwd=str(project_root),
        env=os.environ.copy()
    )
    
    print("\n System running!")
    print("API: http://localhost:8000/docs")
    print("UI:  http://localhost:7860")
    print("\nPress Ctrl+C to stop both services.")
    
    try:
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        print("\n Stopping services...")
        api_process.terminate()
        ui_process.terminate()
        print("Services stopped.")

if __name__ == "__main__":
    main()
