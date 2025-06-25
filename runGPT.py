# OpenAI's GPT API access
import openai
from openai import OpenAI
# Other packages
import time
import os
import re
# Load the synthetic data and prompts
import data

# API key load
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR") + "out\\")

while True:
    try:
        for modelname in [
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-2025-04-14"
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
                                #for temperature in [0.0, 0.1]:
                                for temperature in [0.0]:
                                    # Run everything as triplicates
                                    for rep in range(3):
                                        # Client
                                        client = OpenAI(
                                            api_key=os.environ.get("OPENAI_API_KEY")
                                        )
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
                                            # Run the actual prompt itself
                                            # Both seed and non-seed version
                                            if seed:
                                                response = client.chat.completions.create(
                                                    messages=[
                                                        {
                                                            "role": "user",
                                                            "content": query,
                                                        }
                                                    ],
                                                    model=modelname,
                                                    # Do not allow GPT the extra information in response_format parameter
                                                    # Fair comparison would expect the LLM to infer it from the statement
                                                    #response_format={"type": "json_object"},
                                                    temperature=temperature,
                                                    # Test fixed seed runs on GPT
                                                    seed=1
                                                )
                                            else:
                                                response = client.chat.completions.create(
                                                    messages=[
                                                        {
                                                            "role": "user",
                                                            "content": query,
                                                        }
                                                    ],
                                                    model=modelname,
                                                    # Do not allow GPT the extra information in response_format parameter
                                                    # Fair comparison would expect the LLM to infer it from the statement
                                                    #response_format={"type": "json_object"},
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
                                            time.sleep(1) # Sleep a second
        break
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        time.sleep(120)
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
        time.sleep(120)
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
        time.sleep(120)
