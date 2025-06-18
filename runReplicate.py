# Replicate
import replicate

import os
import time
import data
import re

# API key load
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR") + "out\\")

try:
    for modelname in [
        "meta/llama-4-maverick-instruct",
        "meta/llama-4-scout-instruct",
        "deepseek-ai/deepseek-r1",
        "deepseek-ai/deepseek-v3"
    ]:
        # Prompts to iterate across
        for promptIndex in data.getArrayPromptIndex():
            # Input statements
            for inputIndex in data.getArrayInputIndex():
                # Iterate across languages (0 = English, 1 = Finnish, ...)
                for lang in [0, 1]:
                    # Non-censored (value 0) or censored (1) version of the input statements
                    for cens in [0, 1]:
                        # Seed settings
                        #for seed in [False, True]:
                        for seed in [False]:
                            # Temperature-parameter values
                            for temperature in [0.0, 0.1]:
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
                                        if seed:
                                            input = {
                                                "prompt": query,
                                                "temperature": temperature,
                                                # Test fixed seed runs on Replicate deployed models
                                                "seed": 1
                                            }
                                        else:
                                            input = {
                                                "prompt": query,
                                                "temperature": temperature
                                            }
                                        # Write output to a suitable file
                                        out = ""
                                        retries = 0
                                        while True:
                                            try:
                                                f = open(filename + ".out", 'w', encoding="utf-8")
                                                startTime = time.time()
                                                for iterator in replicate.run(
                                                        modelname,
                                                        input=input
                                                ):
                                                    out = out + "".join(iterator)
                                                    print("".join(iterator), end="", file=f)
                                                endTime = time.time()
                                                print("\nOutput:\n" + out)
                                                f.close()
                                                time.sleep(1)
                                                break
                                            except ValueError as e:
                                                print(
                                                    "Error ( retry " + retries + " ) inside the replicate.run: " + str(
                                                        e))
                                                retries = retries + 1
                                            if retries > 9:
                                                break
                                        # Record wall clock time for the API call (seconds with enough decimals)
                                        f = open(filename + ".time", 'w', encoding="utf-8")
                                        f.write(str(endTime - startTime))
                                        f.close()
                                        # Record the prompt put in (sanity checks etc)
                                        f = open(filename + ".prompt", 'w', encoding="utf-8")
                                        f.write(query)
                                        f.close()
                                        print("\n\n")
except ValueError as e:
    print("Error: " + str(e))
