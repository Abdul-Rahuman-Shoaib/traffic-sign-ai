"""
Vision-Based Traffic Sign Recognition and Road Safety Assistant
Standalone PC Desktop Application (Native GUI Window)
PSN Engineering College, Tirunelveli - Dept. of Computer Science & Engineering
"""
import sys
import os
import time
import threading
import socket
import webview
from app import app, seed_database

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def run_flask_server():
    seed_database()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

def main():
    print("=======================================================================")
    print("Initializing Road Safety Assistant PC Desktop Application...")
    print("PSN Engineering College - Department of Computer Science & Engineering")
    print("=======================================================================")

    # Start Flask server in background daemon thread if not already running
    if not is_port_in_use(5000):
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        time.sleep(1.2) # Allow server to bind

    # Create native PC desktop window using pywebview
    window = webview.create_window(
        title="Vision-Based Traffic Sign Recognition & Road Safety Assistant",
        url="http://127.0.0.1:5000",
        width=1340,
        height=880,
        min_size=(1024, 720),
        text_select=True,
        zoomable=True,
        confirm_close=True,
        background_color="#080c16"
    )

    # Launch native desktop application loop
    webview.start(private_mode=False)

if __name__ == "__main__":
    main()
