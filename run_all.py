# run_all.py
import subprocess
import time

print("Starting FastAPI backend...")
backend = subprocess.Popen(["uvicorn", "src.serving.app:app", "--reload", "--port", "8000"])

time.sleep(3)  # give backend time to start

print("Starting Streamlit dashboard...")
frontend = subprocess.Popen(["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501"])

try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    backend.terminate()
    frontend.terminate()