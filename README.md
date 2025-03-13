Huggingface Model API Server
I've been training a bunch of local models lately (having a great time experimenting!), and I really enjoy using OpenWebUI. However, I couldn't find an easy way to serve Hugging Face models locally with OpenWebUI, similar to how LMStudio handles GGUF models—so I decided to build one.

What it does right now:
Loads Hugging Face models from simple folders (e.g., C:/Models).
Runs a local API endpoint at http://0.0.0.0:5678 (configurable if you prefer another address).
Fully compatible with OpenWebUI’s OpenAI-style connections.
Includes a basic HTML dashboard at the same address for easy loading and unloading of models.
What's coming soon:
Improved GGUF model support.
Enhanced dashboard functionality (currently shows only the last loaded model).
I've tested this setup extensively, and it's working well for my needs—easy deployment, organized setup, and intuitive chat interactions within OpenWebUI.

There's still plenty to polish, but I was excited to share it right away.

If you find this helpful, have suggestions, or know of similar existing tools, please let me know—I’d love your feedback.

Check it out here:  https://github.com/egrigor86/hf_api_server



Below is an AI-generated explanation of the program's code and features:



This is an implementation of an open-source API service for chat functionality using Python's FastAPI framework.

## Prerequisites:
- Python 3.8+
- PyTorch
- transformers library
- llama.cpp (for GGUF models)
- Required model files in the C:/WorkingFolder/Models directory

## Usage:

1. **Starting the Service:**
   ```
   python main.py
   ```

2. **Available Endpoints:**

### Loading Models:
   - POST `/v1/models/load`
     - Payload example:
     ```json
     {
         "model_name": "your_model_name",
         "model_type": "gguf" or "hf"
     }
     ```
   
   - GET `/v1/models` returns available models

3. **Getting Chat Completions:**
   - POST `/v1/chat/completions`
     - Request format:
     ```json
     {
         "model": "your_model_name",
         "messages": [
             {"role": "user", "content": "Your prompt text"},
             ...
         ],
         "temperature": 0.7,
         "top_p": 0.9,
         "repetition_penalty": 1.2
     }
     ```

4. **Model Management:**
   - POST `/v1/models/unload` to remove models from memory

## Features:
- Supports both Llama.cpp (GGUF) and OpenAI-style models (Hugging Face)
- Background processing queue for handling multiple requests
- Model persistence through configuration file
- Support for streaming responses
- Automatic model loading of the last used model on startup

Note: Make sure to have proper error handling in place, as this is a basic implementation that may need additional safeguards for production use.