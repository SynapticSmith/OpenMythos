# OpenMythos: The Complete Beginner-to-Pro Guide

Welcome to **OpenMythos**! If you have absolutely no idea what you are doing, don't worry. This guide is built to hold your hand from the very beginning. We will cover what OpenMythos is, how to set up your computer (even if you have nothing installed), how to understand the confusing jargon, and finally, how to run, train, and connect this model to the real world.

---

## Table of Contents

1. [What is OpenMythos? (Explained Simply)](#1-what-is-openmythos-explained-simply)
2. [Absolute Beginner Setup Guide](#2-absolute-beginner-setup-guide)
   - [Step 1: Install Python](#step-1-install-python)
   - [Step 2: Set up a Virtual Environment](#step-2-set-up-a-virtual-environment)
   - [Step 3: Install PyTorch (CPU or GPU)](#step-3-install-pytorch-cpu-or-gpu)
   - [Step 4: Install OpenMythos](#step-4-install-openmythos)
3. [Understanding the Core Concepts (The Jargon)](#3-understanding-the-core-concepts-the-jargon)
4. [How to Use OpenMythos (Inference)](#4-how-to-use-openmythos-inference)
5. [How to Train OpenMythos](#5-how-to-train-openmythos)
6. [Connecting OpenMythos to the Real World](#6-connecting-openmythos-to-the-real-world)
   - [Example 1: Wrapping in a FastAPI Server](#example-1-wrapping-in-a-fastapi-server)
   - [Example 2: Integration via Model Context Protocol (MCP) / Hermes Harness](#example-2-integration-via-model-context-protocol-mcp--hermes-harness)

---

## 1. What is OpenMythos? (Explained Simply)

OpenMythos is a **Language Model**, similar in concept to ChatGPT or Claude. However, it operates a bit differently under the hood.

Normal AI models process your text by passing it through hundreds of different "layers" sequentially (like an assembly line). OpenMythos is a **Recurrent-Depth Transformer (RDT)**. Instead of hundreds of *different* layers, it has a small set of layers that it **loops through multiple times**.

Imagine you are trying to solve a complex math problem. A normal AI tries to read it once and instantly guess the answer. OpenMythos reads it, loops its reasoning, thinks about it again, loops again, and keeps thinking until it is ready to give an answer. Because it re-uses the same "brain" loops, it can think deeper without taking up massive amounts of memory.

---

## 2. Absolute Beginner Setup Guide

To run OpenMythos, your computer needs a few basic tools installed.

### Step 1: Install Python
Python is the programming language OpenMythos is written in.
1. Go to [python.org/downloads](https://www.python.org/downloads/).
2. Download the latest version of Python 3 (Python 3.10 or higher is required).
3. **Important for Windows Users:** When running the installer, **check the box that says "Add Python to PATH"** before clicking Install.

### Step 2: Set up a Virtual Environment
A virtual environment is like an isolated sandbox. It keeps the files OpenMythos needs separate from the rest of your computer.
1. Open your computer's Terminal (Mac/Linux) or Command Prompt (Windows).
2. Create a new folder for your project and move into it:
   ```bash
   mkdir my_mythos_project
   cd my_mythos_project
   ```
3. Create the virtual environment:
   ```bash
   python -m venv mythos_env
   ```
4. Activate the virtual environment:
   - **Mac/Linux:** `source mythos_env/bin/activate`
   - **Windows:** `mythos_env\Scripts\activate`

   *(You will know it worked if you see `(mythos_env)` at the beginning of your terminal line).*

### Step 3: Install PyTorch (CPU or GPU)
PyTorch is the engine that does all the heavy math for the AI.

**Scenario A: I don't have a GPU (Graphics Card), or I am on a Mac.**
Run this command to install the CPU-only version. It will be slower, but it will work:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Scenario B: I have an NVIDIA GPU (Windows/Linux).**
You can use CUDA to make the AI run much faster.
1. Make sure your NVIDIA drivers are up to date via the NVIDIA website.
2. Run this command:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install OpenMythos
Now that the engine is ready, let's install the car!
```bash
pip install open-mythos
```
*(Alternatively, if you cloned the code from GitHub, you can run `pip install -e .` inside the repository).*

---

## 3. Understanding the Core Concepts (The Jargon)

When you look at OpenMythos code, you'll see a lot of confusing variables. Here is what they mean in plain English:

- **`vocab_size`**: The number of unique "words" (or tokens) the AI knows.
- **`dim`**: The size of the AI's "brain" or hidden memory. Bigger numbers mean a smarter AI, but it requires a more powerful computer.
- **`max_loop_iters`**: This is OpenMythos' superpower. It's the number of times the AI will "loop" over the same thought process before giving an answer. Higher loops = deeper thinking.
- **`n_experts`**: OpenMythos uses a "Mixture of Experts" (MoE). Imagine the AI has 64 different specialists (one for math, one for history, etc.).
- **`n_experts_per_tok`**: How many specialists the AI consults for a specific word. (e.g., if it's 4, it asks 4 specialists).
- **`attn_type`**: How the AI pays attention to your words. "mla" (Multi-Latent Attention) is highly compressed and saves a lot of computer memory. "gqa" (Grouped Query Attention) is an older, more standard method.

OpenMythos has predefined sizes (variants) ready to go, such as:
- **`mythos_1b()`**: Small, fast, can run on weaker computers.
- **`mythos_3b()`**: Mid-sized, good balance of speed and smarts.
- **`mythos_10b()` up to `mythos_1t()`**: Massive models that require industrial supercomputers.

---

## 4. How to Use OpenMythos (Inference)

"Inference" is a fancy word for "asking the AI to generate text."

Create a file named `run.py` and paste the following code:

```python
import torch
from open_mythos import mythos_1b, OpenMythos

# 1. Load the Configuration (we will use the smallest 1 Billion parameter version)
cfg = mythos_1b()

# 2. Build the Model
model = OpenMythos(cfg)

# Optional: If you have an NVIDIA GPU, move the model to the GPU for speed
if torch.cuda.is_available():
    model = model.to("cuda")

# 3. Create some dummy input numbers (representing words)
# In reality, you would use a "Tokenizer" to convert text like "Hello world" into numbers.
dummy_input_ids = torch.randint(0, cfg.vocab_size, (1, 10)) # 1 sentence, 10 words long
if torch.cuda.is_available():
    dummy_input_ids = dummy_input_ids.to("cuda")

# 4. Generate text!
# We tell it to think (loop) 8 times per word, and output a maximum of 20 new words.
output = model.generate(dummy_input_ids, max_new_tokens=20, n_loops=8)

print("Generation Complete! Output shape:", output.shape)
```

Run this file in your terminal:
```bash
python run.py
```

---

## 5. How to Train OpenMythos

Training is how you teach the AI *how* to speak by feeding it massive amounts of text. OpenMythos includes a script to train the `3B` model on the "FineWeb-Edu" dataset (a giant collection of educational websites).

**To run a small test training session on your CPU or single GPU:**
1. Ensure you have the `datasets` and `loguru` packages installed (`pip install datasets loguru`).
2. Run the script:
```bash
python training/3b_fine_web_edu.py
```
*Note: Full training requires immense computing power. Running this on a personal laptop will just be for testing purposes and will run very slowly.*

---

## 6. Connecting OpenMythos to the Real World

Running the model in a script is great, but what if you want to connect it to a website, an app, or an AI Agent system?

### Example 1: Wrapping in a FastAPI Server
You can easily turn OpenMythos into a Web API that other applications can talk to over the internet.

1. Install FastAPI: `pip install fastapi uvicorn`
2. Create a file named `server.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from open_mythos import mythos_1b, OpenMythos

app = FastAPI()

# Setup model
cfg = mythos_1b()
model = OpenMythos(cfg)

class GenerateRequest(BaseModel):
    # For a real app, this would be text. We use dummy ints for this bare-bones example.
    input_numbers: list[int]
    max_tokens: int = 10

@app.post("/generate")
def generate_text(req: GenerateRequest):
    tensor_input = torch.tensor([req.input_numbers])
    output = model.generate(tensor_input, max_new_tokens=req.max_tokens, n_loops=8)
    return {"output_numbers": output[0].tolist()}

# Run this server by typing: uvicorn server:app --reload
```

### Example 2: Integration via Model Context Protocol (MCP) / Hermes Harness
If you are using an Agentic framework like **Hermes Harness** or an **MCP (Model Context Protocol)**, OpenMythos can be packaged as a "Tool" or an "Engine".

Because OpenMythos operates entirely locally in Python, you can wrap it in a standard class interface that MCP expects.

```python
class OpenMythosEngine:
    def __init__(self):
        self.cfg = mythos_1b()
        self.model = OpenMythos(self.cfg)
        # Note: You would also instantiate your Tokenizer here!

    def query(self, prompt: str) -> str:
        # 1. Convert Prompt (Text) to Tokens (Numbers)
        # tokens = self.tokenizer.encode(prompt)
        tokens = torch.randint(0, self.cfg.vocab_size, (1, 10)) # Dummy representation

        # 2. Generate response using the OpenMythos deep-looping generate method
        response_tokens = self.model.generate(tokens, max_new_tokens=50, n_loops=16)

        # 3. Convert Tokens back to Text
        # text = self.tokenizer.decode(response_tokens)
        return "This is the generated text from OpenMythos!"

# In your Hermes Harness or MCP setup, you simply register `OpenMythosEngine.query`
# as the backend logic for answering user questions or executing agent reasoning!
```

By adjusting `n_loops`, you can actually tell the Agent Framework to "think harder" dynamically depending on how complex the user's task is!

---

**You are now ready to explore, build, and experiment with OpenMythos. Happy Coding!**