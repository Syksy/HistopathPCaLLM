# Mistral package used
from mistralai import Mistral
# Other packages
import time
import os
import re
# Load the synthetic data and prompts
import histoPCaData

# API key load
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR") + "out\\")

while True:
    try:
        for modelname in [
            "mistral-large-2411",
            "mistral-medium-2505",
            "mistral-small-2503"
        ]:
            # Prompts to iterate across
            for promptIndex in histoPCaData.getArrayPromptIndex():
                # Iterate across languages (0 = English, 1 = Finnish, ...)
                for lang in [0, 1]:
                    # Input statements
                    for inputIndex in histoPCaData.getArrayInputIndex(lang):
                        # Non-censored (value 0) or censored (1) version of the input statements
                        for cens in [0, 1]:
                            # Seed settings
                            #for seed in [False, True]:
                            for seed in [False]:
                                # Temperature-parameter values
                                #for temperature in [0.0, 0.1]:
                                for temperature in [0.0]:
                                    # Run everything as triplicates
                                    for rep in range(3):
                                        # Client
                                        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
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
                                            query = histoPCaData.getQuery(promptIndex, inputIndex, lang, cens)
                                            print("With query: \n" + query + "\n")
                                            startTime = time.time()
                                            # Run the actual prompt itself
                                            # Both seed and non-seed version
                                            if seed:
                                                response = client.chat.complete(
                                                #response = client.chat.completions.create(
                                                    messages=[
                                                        {
                                                            "role": "user",
                                                            "content": query,
                                                        }
                                                    ],
                                                    model=modelname,
                                                    temperature=temperature,
                                                    # Test fixed seed runs
                                                    seed=1
                                                )
                                            else:
                                                #response = client.chat.completions.create(
                                                response = client.chat.complete(
                                                    messages=[
                                                        {
                                                            "role": "user",
                                                            "content": query,
                                                        }
                                                    ],
                                                    model=modelname,
                                                    temperature=temperature,
                                                )
                                            endTime = time.time()
                                            # Write output to a suitable file, output
                                            f = open(filename + ".out", 'w', encoding="utf-8")
                                            f.write(response.choices[0].message.content)
                                            f.close()
                                            print("\nOutput:\n" + response.choices[0].message.content)
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
                                            time.sleep(5) # Sleep a bit
        break
    except Exception as e:
        print("Error: " + str(e))
        time.sleep(120)
