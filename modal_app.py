"""Modal deployment entrypoint for the FastAPI service. Configure rag-v2-secrets."""
import modal
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("backend/app", remote_path="/root/app")
    .add_local_python_source("backend"))
app = modal.App("hybrid-rag-v2", image=image)

@app.function(secrets=[modal.Secret.from_name("rag-v2-secrets")], cpu=4, memory=8192, scaledown_window=300)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/root")
    from backend.app.main import app as fastapi_app
    return fastapi_app
