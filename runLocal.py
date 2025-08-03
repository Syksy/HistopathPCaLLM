from transformers import pipeline
from huggingface_hub import login
from huggingface_hub import snapshot_download
import data
import time
import re
import requests
# PyTorch installed with CUDA (GPU: RTX 5090)
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
# Late July 2025
import torch
import os

# API key load
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR") + "out\\")

print("Running models locally")
# Checking PyTorch and other parameters:
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

try:
    import triton
    print(f"Triton version: {triton.__version__}")
except ImportError:
    print("Triton not installed")
    torch.backends.cudnn.enabled = False

# Download model snapshot(s)
if False:
    # Login to HuggingFace to gain access to gated medgemma/gemmas/Qwen
    print("Logging in to HuggingFace...")
    login(token=os.environ.get("HUGGINGFACE_API_KEY"))
    print("Logged in to HuggingFace.")

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
        "Qwen/Qwen3-235B-A22B-Instruct-2507"
    ]:
        print("Downloading model snapshot for " + model)
        snapshot_download(
            repo_id=model,
            token=os.environ.get("HUGGINGFACE_API_KEY")
        )
        print("Model snapshot download completed")

# Test running through (med)gemmas locally
for modelname in [
    #"google/gemma-3-4b-it",
    #"google/gemma-3-12b-it",
    #"google/gemma-3-27b-it",
    #"google/medgemma-4b-it",
    "google/medgemma-27b-it",
    #"google/medgemma-27b-text-it",
]:
    print("Creating pipeline...")
    pipe = pipeline(
        # "image-text-to-text",
        task="text-generation",
        model=modelname,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    # Prompts to iterate across
    for promptIndex in data.getArrayPromptIndex():
        # Input statements
        for inputIndex in data.getArrayInputIndex():
            # Iterate across languages (0 = English, 1 = Finnish, ...)
            for lang in [0, 1]:
                # Non-censored (value 0) or censored (1) version of the input statements
                for cens in [0, 1]:
                    # Seed settings
                    # for seed in [False, True]:
                    for seed in [False]:
                        # Temperature-parameter values
                        # for temperature in [0.0, 0.1]:
                        for temperature in [0.0]:
                            # Run everything as triplicates
                            for rep in range(3):
                                # Construct file name
                                filename = ("HistopathPCaLLM_" + re.sub("/", "-", modelname)
                                            + "_prompt" + str(promptIndex)
                                            + "_input" + str(inputIndex)
                                            + "_lang" + str(lang)
                                            + "_cens" + str(cens)
                                            + "_seed" + str(seed)
                                            + "_temp" + str(temperature)
                                            + "_rep" + str(rep)
                                            )
                                # Run only if the output file doesn't exist yet
                                if not os.path.isfile(os.path.realpath(filename + ".out")):
                                    print("Running \n" + filename + "\n")
                                    # Construct the prompt + statement query to send
                                    query = data.getQuery(promptIndex, inputIndex, lang, cens)
                                    print("With query: \n" + query + "\n")
                                    startTime = time.time()

                                    output = pipe(text_inputs=
                                        [
                                            {
                                                "role": "user",
                                                "content": [
                                                    {"type": "text", "text": query}
                                                ]
                                            }
                                        ],
                                    #max_new_tokens=2048)
                                    max_new_tokens=200)
                                    response = output[0]["generated_text"][-1]["content"]
                                    endTime = time.time()
                                    # Write output to a suitable file, output
                                    #f = open(filename + ".out", 'w', encoding="utf-8")
                                    #f.write(response)
                                    #f.close()
                                    print("\nOutput:\n" + response)
                                    # Record wall clock time for the API call (seconds with enough decimals)
                                    #f = open(filename + ".time", 'w', encoding="utf-8")
                                    #f.write(str(endTime - startTime))
                                    #f.close()
                                    # Record the prompt put in (sanity checks etc)
                                    #f = open(filename + ".prompt", 'w', encoding="utf-8")
                                    #f.write(query)
                                    #f.close()
                                    # Handle console output
                                    print("\n\n")
    # Clear GPU memory
    del pipe
    torch.cuda.empty_cache()
