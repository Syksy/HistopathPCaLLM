# OpenAI's GPT API access
import openai
from openai import OpenAI
# Other packages
import time
import os
import re
# API key load
from dotenv import load_dotenv
load_dotenv()

# Load the synthetic data and prompts
import data

# Load OpenAI API key from local environment; starts with "sk-proj-..."
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR") + "out\\")

try:
    for modelname in [
        #"gpt-3.5-turbo-0125",
        #"gpt-3.5-turbo-1106",
        #"gpt-4-0613",
        #"gpt-4-0125-preview", # Same as "gpt-4-turbo-preview"
        #"gpt-4-turbo-2024-04-09",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        # "gpt-4o-latest-0326", # Exists?
        # "gpt-4o-mini-2024-07-18",
        # "o1-mini-2024-09-12",
        # "o1-2024-12-17"
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
                        for seed in [False, True]:
                            # Temperature-parameter values
                            for temperature in [0.0, 0.1, 0.3]:
                                # Run everything as triplicates
                                for rep in range(3):
                                    # Construct file name
                                    filename = ("HistopathPcaLLM_" + modelname
                                                + "_prompt" + str(promptIndex)
                                                + "_input" + str(inputIndex)
                                                + "_lang" + str(lang)
                                                + "_cens" + str(cens)
                                                + "_seed" + str(seed)
                                                + "_temp" + str(temperature)
                                                + "_rep" + str(rep)
                                    )


except openai.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except openai.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except openai.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)
