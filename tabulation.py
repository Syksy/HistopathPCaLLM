# Tabulate and collect results in a suitable format across all output files
# Use numpy multidimensional arrays
import os
import data
import numpy as np
import re
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR"))

# Allow debug output
debug = False
# Allow progress report output
progress = True

# Names of explored models
modelnames = [
    # Claudes
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    # Geminis
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    #"gemini-1.5-pro-001",
    #"gemini-1.5-pro-002",
    # OpenAI / GPTs
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-2025-04-14",
    # Grok
    "grok-3-beta",
    "grok-2-1212",
    # Replicate runs
    "meta-llama-4-maverick-instruct",
    "meta-llama-4-scout-instruct",
    "deepseek-ai-deepseek-r1",
    "deepseek-ai-deepseek-v3",
    "google-deepmind-gemma-3-4b-it",
    "google-deepmind-gemma-3-12b-it",
    "google-deepmind-gemma-3-27b-it",
    # Mistral
    "mistral-large-2411",
    "mistral-medium-2505",
    "mistral-small-2503",
]

# Full response returned by LLM
full = np.chararray(
    shape=(
        # Censored False/True
        2,
        # Seed False/True
        2,
        # 3 repeats
        3,
        # Number of tested models
        len(modelnames),
        # Maximum number of run prompt variation wrapping the statement
        data.getMaxPromptLength(),
        # Maximum number of statements
        data.getMaxInputLength(),
        # Number of languages (0 = English, 1 = Finnish, 2 = Swedish)
        #3
        2
    ),
    itemsize=2000
)
# Character output returned by LLM
output = np.chararray(
    shape=(
        # Censored False/True
        2,
        # Seed False/True
        2,
        # 3 repeats
        3,
        # Number of tested models
        len(modelnames),
        # Maximum number of run prompt variation wrapping the statement
        data.getMaxPromptLength(),
        # Maximum number of statements
        data.getMaxInputLength(),
        # Number of languages (0 = English, 1 = Finnish, 2 = Swedish)
        #3
        2
    ),
    itemsize=1000
)
# Runtime (seconds with decimals)
runtimes = np.ndarray(
    shape=(
        # Censored False/True
        2,
        # Seed False/True
        2,
        # 3 repeats
        3,
        # Number of tested models
        len(modelnames),
        # Maximum number of run prompt variation wrapping the statement
        data.getMaxPromptLength(),
        # Maximum number of statements
        data.getMaxInputLength(),
        # Number of languages (0 = English, 1 = Finnish, 2 = Swedish)
        #3
        2
    ),
    dtype='f'
)

filesSuccess = 0
filesFail = 0

# Signal containing statement (0) or censored (1)
for cens in range(2):
    # Some models allowed adding RNG seeding for reproducibility
    #for seed in range(2):
    for seed in [0]:
        # Iterate over models
        for modelIndex in range(len(modelnames)):
            # Iterate over prompts
            for promptIndex in data.getArrayPromptIndex():
                # Iterate over inputs
                for inputIndex in data.getArrayInputIndex():
                    # Iterate over replicates
                    for rep in range(3):
                        # Iterate across languages (0 = English, 1 = Finnish, 2 = Swedish)
                        for langIndex in [0, 1]:
                            # String name for seed True/False
                            seedname = str([False, True][seed])
                            # Construct file name
                            filename = ("out\\HistopathPCaLLM_" + modelnames[modelIndex]
                                        + "_prompt" + str(promptIndex)
                                        + "_input" + str(inputIndex)
                                        + "_lang" + str(langIndex)
                                        + "_cens" + str(cens)
                                        + "_seed" + str(seedname)
                                        #+ "_temp" + str(temperature)
                                        + "_temp" + str(0.0)
                                        + "_rep" + str(rep)
                                        )

                            if os.path.isfile(filename + ".out"):
                                file = open(filename + ".out", 'r', encoding="utf-8")
                                lines = file.readlines()
                                # Full output prior to any potential parsing
                                full[cens, seed, rep, modelIndex, promptIndex, inputIndex, langIndex] = "".join(lines).strip().encode("utf-8")

                                start = 0
                                end = len(lines)
                                # Sanitize; extract {JSON} part from  ...```json {JSON} -- ```...
                                # or ... ``` {JSON} ```
                                for i in range(len(lines)):
                                    if lines[i].strip() == "```json" or lines[i].strip() == "```" \
                                            or lines[i].strip() == "[```json]" or lines[i].strip() == "[" \
                                            or re.search(r"```json$", lines[i].strip()): # Allow ending to ANYTHING```json as a re-prefix (mainly for GPT-OSS)
                                        start = i+1
                                        break
                                    # Alternatively there may be comments pre/post a proper { <json> } content, limit to that
                                    elif lines[i].strip() == "{":
                                        start = i
                                        break
                                # The ending to ``` sequence if one was detected; starting sequence from prior start point
                                for j in range(start, len(lines)):
                                    if lines[j].strip() == "```" or lines[j].strip() == "]" \
                                            or lines[j].strip() == "```<end_of_turn>":
                                        end = j
                                        break
                                    # Alternatively there may be comments pre/post a proper { <json> } content, limit to that
                                    elif lines[j].strip() == "}":
                                        end = j+1
                                        break
                                # Prune end if we try to grab a null line at the end
                                if end > len(lines):
                                    end = len(lines)-1
                                # Grab the JSON part
                                lines = lines[start:end]
                                lines = "".join(lines).strip()
                                file.close()
                                filesSuccess = filesSuccess + 1
                            else:
                                print("Could not find: " + filename)
                                filesFail = filesFail + 1
                                lines = "<NA>"

                            if debug:
                                print("-- Found output content:\n" + lines + "\n--\n")
                                print("Finished rep " + str(rep) + " modelIndex " + str(modelIndex) + " promptIndex "
                                  + str(promptIndex) + " inputIndex " + str(inputIndex) + " lang " + str(langIndex) +
                                  + " seed " + str(seedname))

                            output[cens, seed, rep, modelIndex, promptIndex, inputIndex, langIndex] = lines.encode("utf-8")
                            if os.path.isfile(filename + ".time"):
                                file = open(filename + ".time", 'r', encoding="utf-8")
                                lines = file.readlines()
                                lines = "".join(lines).strip()
                                file.close()
                            else:
                                # If there is an error reading the runtime, the value is parsed as "-1 seconds"
                                lines = 'nan'
                            if debug:
                                print("-- Found runtime content:\n" + lines + "\n--\n")
                            try:
                                runtimes[cens, seed, rep, modelIndex, promptIndex, inputIndex, langIndex] = float(lines.encode("utf-8"))
                            except ValueError as e:
                                # Numeric conversion error for runtimes (likely empty file); marking errors as -1
                                runtimes[cens, seed, rep, modelIndex, promptIndex, inputIndex, langIndex] = float('nan')

if progress:
    print("\n\nExample full: \n")
    print(output[0,0,0,0,0,0].decode("utf-8"))
    print("\n\nExample sanitized output: \n")
    print(output[0,0,0,0,0,0].decode("utf-8"))
    print("\n\nExample runtime: \n")
    print(runtimes[0,0,0,0,0,0])
    print("\n\nDimensions and shape of output: \n")
    print(str(output.ndim) + "\n")
    print(str(output.shape) + "\n")
    print("Number of successfully processed filenames:" + str(filesSuccess))
    print("Number of unsuccessfully processed filenames:" + str(filesFail))
# Write out the full 6-dim np char array as binary file
file = open("npydata\\full.npy", 'wb')
np.save(file=file, arr=full)
file.close()
# Write out the sanitized 6-dim np char array as binary file
file = open("npydata\\output.npy", 'wb')
np.save(file=file, arr=output)
file.close()
# Write out runtimes floats as a binary file
file = open("npydata\\runtimes.npy", 'wb')
np.save(file=file, arr=runtimes)
file.close()
