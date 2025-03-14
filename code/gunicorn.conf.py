bind = "0.0.0.0:8000"
workers = 1  # Gradio works best with 1 worker
timeout = 600  # Increased timeout for long-running operations
worker_class = "gthread"
threads = 4