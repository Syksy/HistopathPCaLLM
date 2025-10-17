import anthropic
import os

import histoPCaData
import time

# Environmental variables
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR") + "out\\")

# Latest models taken from https://docs.anthropic.com/en/docs/resources/model-deprecations#model-status
# Loop while there is still combinations left to run (i.e. overloaded error or similar, sleep it off)
while True:
    try:
        for modelname in [
            # Claude 3.5 Haiku
            "claude-3-5-haiku-20241022",
            # Claude 3.5/3.7 Sonnets
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            # Claude 4s
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            # Claude 4.1s
            "claude-opus-4-1-20250805",
        ]:
            # Prompts to iterate across
            for promptIndex in histoPCaData.getArrayPromptIndex():
                # Iterate across languages (0 = English, 1 = Finnish, 2 = Swedish)
                for lang in [0, 1, 2]:
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
                                        client = anthropic.Anthropic(
                                            api_key=os.environ.get("ANTHROPIC_API_TOKEN"),
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
                                            query = histoPCaData.getQuery(promptIndex, inputIndex, lang, cens)
                                            print("With query: \n" + query + "\n")
                                            startTime = time.time()
                                            # Run the actual prompt itself
                                            message = client.messages.create(
                                                model=modelname,
                                                max_tokens=5000,
                                                temperature=temperature,
                                                messages=[
                                                    {
                                                        "role": "user",
                                                        "content": [
                                                            {
                                                                "type": "text",
                                                                "text": query
                                                            }
                                                        ]
                                                    }
                                                ]
                                            )
                                            endTime = time.time()
                                            # Claude sometimes refused to answer, citing Usage Policy
                                            if len(message.content) == 0 and message.stop_reason == "refusal":
                                                response = "Refusal to answer"
                                            else:
                                                response = "".join(message.content[0].text)
                                            # Write output to a suitable file, output
                                            f = open(filename + ".out", 'w', encoding="utf-8")
                                            f.write(response)
                                            f.close()
                                            print("\nOutput:\n" + response)
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
                                            time.sleep(5)  # Sleep a bit
        break
    except anthropic.InternalServerError as e:
        print("Error: " + str(e))
        time.sleep(120)
    except Exception as e:
        print("Error: " + str(e))
        time.sleep(120)
