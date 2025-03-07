import os
from typing import List, Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
from pydantic import BaseModel
import json
from prompt_templates import SPECIALS_PROMPT
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Restaurant Specials Generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (update for production)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration for AI API
WEBUI_ENABLED = True
WEBUI_BASE_URL = "https://chat.ivislabs.in/api"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjhhM2FjZDUyLTJiZmQtNDQwOS1hMThlLTNhYjIxYzMxZGZlYyJ9.vwFa3YAG9pL3H5taMU59FwKSXohtoNwnDZTeTFGjwBQ"  # Replace with actual API key
DEFAULT_MODEL = "gemma2:2b"

class SpecialsRequest(BaseModel):
    ingredients: str
    cuisine: str
    num_specials: int = 3

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_specials(
    ingredients: str = Form(...),
    cuisine: str = Form(...),
    num_specials: int = Form(3),
    model: str = Form(DEFAULT_MODEL)
):
    try:
        prompt = SPECIALS_PROMPT.format(
            ingredients=ingredients,
            cuisine=cuisine,
            num_specials=num_specials
        )
        
        messages = [{"role": "user", "content": prompt}]
        request_payload = {"model": model, "messages": messages}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{WEBUI_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json=request_payload,
                timeout=60.0
            )
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"generated_specials": generated_text}

        raise HTTPException(status_code=500, detail="Failed to generate specials")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating specials: {str(e)}")

@app.get("/models")
async def get_models():
    try:
        if WEBUI_ENABLED:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{WEBUI_BASE_URL}/models",
                        headers={"Authorization": f"Bearer {API_KEY}"}
                    )
                    if response.status_code == 200:
                        models_data = response.json()
                        model_names = [model["id"] for model in models_data.get("data", []) if "id" in model]
                        if model_names:
                            return {"models": model_names}
            except Exception as e:
                print(f"Error fetching models from open-webui API: {str(e)}")

        fallback_models = [DEFAULT_MODEL, "gemma2:2b", "qwen2.5:0.5b", "deepseek-r1:1.5b", "deepseek-coder:latest"]
        return {"models": fallback_models}
    except Exception as e:
        print(f"Unexpected error in get_models: {str(e)}")
        return {"models": [DEFAULT_MODEL]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8880, reload=True)