from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from queue import Queue
import threading
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
from typing import Dict, Optional, List
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import uuid
import json

app = FastAPI()

request_queue = Queue()

# Store loaded models
models: Dict[str, object] = {}
current_model: Optional[str] = None  # Track the currently loaded model

MODEL_DIR = "C:/WorkingFolder/Models"

CONFIG_FILE = "config.json"

def save_current_model():
    """Save the currently loaded model to config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump({"current_model": current_model}, f)

def load_last_model():
    """Load the last used model from config.json if available."""
    global current_model
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                last_model = config.get("current_model")
                if last_model:
                    print(f"🔄 Auto-loading last used model: {last_model}")
                    load_model(ModelLoadRequest(model_name=last_model, model_type="hf"))  # Assuming HF model as default
        except Exception as e:
            print(f"⚠️ Error loading config.json: {e}")


def list_available_models():
    """List all models in the MODEL_DIR"""
    return [f for f in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, f))]

class ModelLoadRequest(BaseModel):
    model_name: str
    model_type: str  # 'gguf' or 'hf'

class ChatCompletionRequest(BaseModel):
    model: str  # Match LM Studio’s API format
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2


def request_worker():
    """Continuously processes requests in the queue."""
    while True:
        request_data, response_queue = request_queue.get()
        process_request(request_data, response_queue)
        request_queue.task_done()

threading.Thread(target=request_worker, daemon=True).start()

# Start the request processing thread
threading.Thread(target=request_worker, daemon=True).start()

# Start the request processing thread
threading.Thread(target=request_worker, daemon=True).start()

def process_request(request_data, response_queue):
    global current_model

    model_name = request_data["model"]
    messages = request_data["messages"]
    temperature = request_data.get("temperature", 0.7)
    top_p = request_data.get("top_p", 0.9)
    repetition_penalty = request_data.get("repetition_penalty", 1.2)

    if model_name != current_model:
        load_model(ModelLoadRequest(model_name=model_name, model_type="hf"))

    if current_model not in models:
        response_queue.put(JSONResponse(content={"error": "Model not loaded"}, status_code=404))
        return

    # Detect if autocomplete (OpenWebUI sends these specially)
    is_autocomplete = any("Task" in msg["content"] for msg in messages)

    user_input = messages[-1]["content"]

    if is_autocomplete:
        import re
        match = re.search(r"<text>(.*?)</text>", user_input, re.DOTALL)
        if match:
            user_input = match.group(1).strip()

    model_data = models[current_model]

    if isinstance(model_data, Llama):  # GGUF Model
        prompt = f"User: {user_input}"
        output = model_data(prompt, max_tokens=2048, temperature=temperature, top_p=top_p, repeat_penalty=repetition_penalty)
        response_text = output["choices"][0]["text"]
    else:  # Hugging Face Model
        tokenizer, model = model_data
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chat_prompt = f"User: {user_input}"
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=256, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=True)
        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if is_autocomplete:
        response = {"text": response_text.strip()}
    else:
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text.strip()},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs["input_ids"][0]),
                "completion_tokens": len(output_ids[0]),
                "total_tokens": len(inputs["input_ids"][0]) + len(output_ids[0])
            }
        }

    response_queue.put(JSONResponse(content=response, status_code=200))

@app.post("/v1/chat/completions")
async def chat_completion(request: Request):
    body = await request.json()
    response_queue = Queue()
    request_queue.put((body, response_queue))

    # Wait until response is ready
    response = response_queue.get()
    return response

async def generate_response_stream(model, tokenizer, prompt, temperature, top_p, repetition_penalty):
    """Generate text token-by-token for streaming output."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
    output_generator = model.generate(
        input_ids,
        max_new_tokens=256,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True
    )

    for output_ids in output_generator.sequences:
        response_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        yield f"data: {response_text.strip()}\n\n"

@app.get("/v1/models")
def get_models():
    available_models = list_available_models()
    return {
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "organization_owner"
            }
            for model_name in available_models
        ]
    }

@app.post("/v1/models/load")
def load_model(request: ModelLoadRequest):
    global current_model

    # If the model is already loaded, skip reloading
    if request.model_name == current_model:
        return {"message": f"Model {request.model_name} is already active."}

    model_path = os.path.abspath(os.path.join(MODEL_DIR, request.model_name))

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    try:
        if request.model_type == "gguf":
            models[request.model_name] = Llama(model_path=model_path)
        elif request.model_type == "hf":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            models[request.model_name] = (tokenizer, model)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    current_model = request.model_name
    save_current_model()  # Save the current model to config.json
    return {"message": f"Model {request.model_name} loaded and selected successfully."}

@app.post("/v1/models/unload")
def unload_model(model_name: str):
    global current_model
    if model_name in models:
        del models[model_name]
        if model_name == current_model:
            current_model = None
        return {"message": f"Model {model_name} unloaded successfully."}
    raise HTTPException(status_code=404, detail="Model not loaded")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    available_models = list_available_models()

    loaded_model_display = current_model if current_model else "None loaded"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Local Hugging Face Model Dashboard</title>
        <script>
            async function loadModel() {{
                const model_name = document.getElementById('model_select').value;
                const model_type = document.getElementById('model_type').value;

                const res = await fetch('/v1/models/load', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{model_name, model_type}})
                }});
                const data = await res.json();
                alert(data.message || data.detail);
                location.reload();
            }}

            async function unloadModel() {{
                const model_name = document.getElementById('model_select').value;

                const res = await fetch(`/v1/models/unload?model_name=${{model_name}}`, {{
                    method: 'POST'
                }});
                const data = await res.json();
                alert(data.message || data.detail);
                location.reload();
            }}

            async function sendChat() {{
                const prompt = document.getElementById('prompt').value;
                const chat_output = document.getElementById('chat_output');

                const res = await fetch('/v1/chat/completions', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        model: "{current_model}",
                        messages: [{{role: "user", content: prompt}}]
                    }})
                }});
                const data = await res.json();
                if(data.choices){{
                    chat_output.innerHTML += "<b>You:</b> " + prompt + "<br><b>Bot:</b> " + data.choices[0].message.content + "<hr>";
                }} else {{
                    chat_output.innerHTML += "<b>Error:</b> " + (data.detail || 'Unknown error') + "<hr>";
                }}
                document.getElementById('prompt').value = "";
            }}
        </script>
    </head>
    <body>
        <h1>🚀 Local HF Model Dashboard</h1>

        <h3>Currently Loaded Model: <span style="color:blue;">{loaded_model_display}</span></h3>

        <h2>Model Management</h2>
        <select id="model_select">
            {''.join([f'<option>{m}</option>' for m in available_models])}
        </select>

        <select id="model_type">
            <option value="hf">HF (HuggingFace)</option>
            <option value="gguf">GGUF</option>
        </select>

        <button onclick="loadModel()">Load Model</button>
        <button onclick="unloadModel()">Unload Model</button>

        <h2>Chat with Model</h2>
        {"<input id='prompt' type='text' placeholder='Type a message...' style='width:300px;'/><button onclick='sendChat()'>Send</button>" if current_model else "<b>No model loaded yet.</b>"}
        <div id="chat_output" style="margin-top:20px; border:1px solid #ddd; padding:10px; width:500px; height:300px; overflow:auto;"></div>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    load_last_model()  # 🔄 Load last model on startup
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5678)
