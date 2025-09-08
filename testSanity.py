# Sanity checking and tests of prompts, input synthetic statements etc
import os
import time
import re
# Running models on Replicate
import replicate
# Import synthetic data and prompts
import histoPCaData
# Testing schema
import json
from jsonschema import validate, ValidationError

# API key load
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.path.join(os.environ.get("ROOT_DIR"), "sanity"))

# Prompt to use as sanity checking whether there's signal or not
prompt = "You will be provided by a statement, which is a prostate cancer histopathology statement in English, Finnish, or Swedish. You should assess whether the statement is informative in that it provides enough expert information to reliably assess what was the origin of the sample (biopsy, turp treatment, radical prostatectomy, or some other sample origin), whether there is a malignant finding in the statement (consider Gleason 3+3=6 and above to be malignant), and if malignancy can be reliable assessed, if its Gleason can be inferred (in format major+minor=sum).\n\nAssess the following statement:\n\'"

while True:
    try:
        for modelname in [
            "moonshotai/kimi-k2-instruct",
            "openai/o1",
        ]:
            # Iterate across languages (0 = English, 1 = Finnish, 2 = Swedish)
            for lang in [0, 1, 2]:
                # Input statements
                for inputIndex in histoPCaData.getArrayInputIndex():
                    # Non-censored (value 0) or censored (1) version of the input statements
                    for cens in [0, 1]:
                        # Construct filename
                        filename = ("SanityCheck_" + re.sub("/", "-", modelname)
                                    + "_input" + str(inputIndex)
                                    + "_lang" + str(lang)
                                    + "_cens" + str(cens)
                                    )
                        # Fetch statement
                        statement = histoPCaData.getStatement(inputIndex, lang, cens)
                        # Some statements are still missing from e.g. lang == 2, omitting these empty strings
                        # Also check the file does not already exist
                        if not statement == "" and not os.path.isfile(os.path.realpath(filename + ".check")):
                            out = ""
                            # Instructions provided in the prompt followed by the statement to be sanity checked
                            query = prompt + statement + '\''
                            print("Query going in:\n\n" + query)
                            f = open(filename + ".check", 'w', encoding="utf-8")
                            for iterator in replicate.run(
                                    modelname,
                                    input={
                                        "prompt": query
                                    }
                            ):
                                out = out + "".join(iterator)
                                print("".join(iterator), end="", file=f)
                            print("Output:\n\n" + out)
        break
    except Exception as e:
        print("Error: " + str(e))
        time.sleep(120)
