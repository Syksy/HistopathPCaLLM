# Google Gemini API access
# Refs
# EU does not allow Gemini API, has to be done inside Google Cloud via Vertex AI
# Below is the code used there-in
# First needed in the cloud shell:
# pip3 install "google-cloud-aiplatform>=1.38"
# python -m pip install python-dotenv

import os
#from google import genai
#import google.generativeai as genai
import data
import time
#import vertexai
#from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold, FinishReason
from openai import OpenAI

# Environmental variables
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_TOKEN"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
#client = genai.Client(api_key="GEMINI_API_TOKEN")
#genai.configure(api_key="GEMINI_API_TOKEN")
#vertexai.init(project=os.environ.get("GEMINI_PROJECT"), location="europe-north1")
#generation_config = GenerationConfig(
#    temperature=0.0
#)
# Trying to remove effect of safety filters
#safety_settings = {
#    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
#}

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR") + "out\\")

try:
    for modelname in [
            #"gemini-2.5-pro-preview-03-25",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-1.5-pro-002"
        ]:
        # Prompts to iterate across
        for promptIndex in data.getArrayPromptIndex():
            # Input statements
            for inputIndex in data.getArrayInputIndex():
                # Iterate across languages (0 = English, 1 = Finnish, ...)
                for lang in [0]:
                    # Non-censored (value 0) or censored (1) version of the input statements
                    for cens in [0, 1]:
                        # Seed settings
                        # for seed in [False, True]:
                        for seed in [False]:
                            # Temperature-parameter values
                            #for temperature in [0.0, 0.1]:
                            for temperature in [0.0]:
                                # Run everything as triplicates
                                for rep in range(3):
                                    # Construct file name
                                    filename = ("HistopathPCaLLM_" + modelname
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
                                        response = client.chat.completions.create(
                                            model=modelname,
                                            n=1,
                                            messages=[
                                                {
                                                    "role": "user",
                                                    "content": query,
                                                }
                                            ],
                                            temperature=temperature
                                        )
                                        #model = GenerativeModel(modelname)
                                        #retries = 1
                                        #while True:
                                        #    startTime = time.time()
                                        #    response = model.generate_content(
                                        #        query,
                                        #        generation_config=generation_config,
                                        #        #safety_settings=safety_settings,
                                        #        stream=False
                                        #    )
                                        #    endTime = time.time()
                                        #    # A lot of prompts are getting caught in the safety blocks even when they've been set to NONE? Try iterate
                                        #    # FinishReason.STOP is the response to a successful run through
                                        #    if response.candidates[0].finish_reason == FinishReason.STOP:
                                        #        break
                                        #    # Run failed, retrying
                                        #    # Appears to be an issue particularly in gemini-1.0-pro-002 ... fallback to older release
                                        #    # even with safety_settings put in
                                        #    else:
                                        #        print("Retrying, caught in harm filters likely FinishReason.OTHER... retry index " + str(retries))
                                        #        if retries > 9:
                                        #            break
                                        #        retries = retries + 1
                                        #        time.sleep(11)
                                        endTime = time.time()
                                        # Write output to a suitable file, output
                                        f = open(filename + ".out", 'w', encoding="utf-8")
                                        #if retries < 10:
                                        #f.write(response.text)
                                        f.write(response.choices[0].message.content)
                                        #f.close()
                                        # Record wall clock time for the API call (seconds with enough decimals)
                                        f = open(filename + ".time", 'w', encoding="utf-8")
                                        f.write(str(endTime - startTime))
                                        f.close()
                                        # Record the prompt put in (sanity checks etc)
                                        f = open(filename + ".prompt", 'w', encoding="utf-8")
                                        f.write(query)
                                        f.close()
                                        # Handle console output and sleep before next API call
                                        print("\n\n")
                                        time.sleep(2)
except ValueError as e:
    print("Error: " + e)
