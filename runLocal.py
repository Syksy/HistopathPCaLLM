from transformers import pipeline
from PIL import Image
from huggingface_hub import login
from huggingface_hub import snapshot_download
import requests
# PyTorch installed with CUDA (GPU: RTX 5090)
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
# Late July 2025
import torch
import os

# API key load
from dotenv import load_dotenv
load_dotenv()

# Login to HuggingFace to gain access to gated medgemma/gemmas
print("Logging in...")
login(token=os.environ.get("HUGGINGFACE_API_KEY"))
print("Logged in.")

# Download model snapshot(s)
for model in [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/medgemma-4b-it",
    "google/medgemma-27b-it",
    "google/medgemma-27b-text-it",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
]:
    print("Downloading model snapshot for " + model)
    snapshot_download(
        repo_id=model,
        token=os.environ.get("HUGGINGFACE_API_KEY")
    )
    print("Model snapshot download completed")

print("Creating pipeline...")
pipe = pipeline(
    "image-text-to-text",
    #"text-generation",
    #model="google/medgemma-4b-it",
    model="google/medgemma-27b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
print("Created pipeline.")

print("Construct call and run it.")
messages = [
    #{
    #    "role": "system",
    #    "content": [{"type": "text", "text": "You are an expert radiologist."}]
    #},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the typical symptoms of prostate cancer."}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
