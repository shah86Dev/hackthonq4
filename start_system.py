"""
Startup script for the Book-Embedded RAG Chatbot system
"""
import os
import subprocess
import sys
import threading
import time
import requests
import argparse
from pathlib import Path


def check_backend_health():
    """Check if the backend is running and healthy"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_backend():
    """Start the backend server"""
    print("Starting backend server...")
    try:
        # Change to backend directory and start the server
        os.chdir("backend")
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        os.chdir("..")  # Return to main directory
        return process
    except Exception as e:
        print(f"Error starting backend: {e}")
        return None


def start_frontend():
    """Start the frontend server"""
    print("Starting frontend server...")
    try:
        # Change to frontend directory and start the server
        os.chdir("frontend")
        process = subprocess.Popen([
            "npx", "docusaurus", "start"
        ])
        os.chdir("..")  # Return to main directory
        return process
    except Exception as e:
        print(f"Error starting frontend: {e}")
        return None


def install_dependencies():
    """Install required dependencies"""
    print("Installing backend dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"], check=True)
        print("Backend dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("Error installing backend dependencies")
        return False

    print("Installing frontend dependencies...")
    try:
        os.chdir("frontend")
        subprocess.run(["npm", "install"], check=True)
        os.chdir("..")
        print("Frontend dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("Error installing frontend dependencies")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Book-Embedded RAG Chatbot System")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--backend-only", action="store_true", help="Start only the backend")
    parser.add_argument("--frontend-only", action="store_true", help="Start only the frontend")
    parser.add_argument("--test", action="store_true", help="Run system tests")

    args = parser.parse_args()

    if args.install:
        if install_dependencies():
            print("\nDependencies installed successfully!")
        else:
            print("\nFailed to install dependencies!")
            return

    if args.test:
        import test_rag_system
        test_rag_system.run_comprehensive_test()
        return

    processes = []

    if not args.frontend_only:
        backend_process = start_backend()
        if backend_process:
            processes.append(("Backend", backend_process))
            print("Backend started on http://localhost:8000")
        else:
            print("Failed to start backend")
            return

        # Wait a bit for the backend to start
        print("Waiting for backend to start...")
        time.sleep(5)

        # Check if backend is healthy
        if check_backend_health():
            print("✓ Backend is healthy")
        else:
            print("✗ Backend is not responding")
            return

    if not args.backend_only:
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(("Frontend", frontend_process))
            print("Frontend started on http://localhost:3000")
        else:
            print("Failed to start frontend")

    print("\nSystem started! Press Ctrl+C to stop.")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:3000")
    print("API Documentation: http://localhost:8000/api/docs")

    try:
        # Wait for all processes to complete
        for name, process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\nShutting down processes...")
        for name, process in processes:
            process.terminate()
            process.wait()
        print("All processes stopped.")


if __name__ == "__main__":
    main()