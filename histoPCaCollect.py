# Collect and process results
import os
import numpy as np
import json
import re
import pandas as pd
import time
# Histopath data and prompts
import histoPCaData

# .env vars load
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR"))

## Functions for assisting in inspecting run differences

# Return True if triplicates are exactly concordant, False otherwise
def getConcordanceExact(rep0 : str, rep1 : str, rep2 : str) -> bool:
    if rep0 == rep1 and rep0 == rep2 and rep1 == rep2:
        return True
    else:
        return False


# Return True if triplicates are concordant after ignoring character case (lower/upper)
def getConcordanceCaseinsensitive(rep0 : str, rep1 : str, rep2 : str) -> bool:
    if rep0.lower() == rep1.lower() and rep0.lower() == rep2.lower() and rep1.lower() == rep2.lower():
        return True
    else:
        return False

# Test if the JSON parseable content is constant across triplicates
# That is, checking that the JSON returned contains the same value in each field across triplicates
def getConcordanceContent(rep0: str, rep1: str, rep2: str) -> bool:
    # If even one of the triplicates cannot be parsed to JSON then the JSON content is not concordant
    if (not getJSONParseability(rep0)) or (not getJSONParseability(rep1)) or (not getJSONParseability(rep2)):
        return False
    rep0json = list(json.loads(rep0).values())
    rep1json = list(json.loads(rep1).values())
    rep2json = list(json.loads(rep2).values())
    # Correct length ought to be 3 values
    if (not (len(rep0json) == 3)) or (not (len(rep1json) == 3)) or (not (len(rep2json) == 3)):
        return False
    # Checking that contents match within index
    for index in range(3):
        if (not (rep0json[index] == rep1json[index])) or (not (rep1json[index] == rep2json[index])):
            return False
    # All comparisons were concordant; returning True
    return True


# Return maximum character count different across triplicates
def getMedianCharCount(rep0 : str, rep1 : str, rep2 : str) -> int:
    return sorted([len(rep0), len(rep1), len(rep2)])[1]

# Return maximum character count different across triplicates
def getMaxCharDiff(rep0 : str, rep1 : str, rep2 : str) -> int:
    return max(abs(len(rep0)-len(rep1)), abs(len(rep0)-len(rep2)), abs(len(rep1)-len(rep2)))


# Returns largest float difference between runtimes
def getMaxRuntimeDiff(rep0: float, rep1: float, rep2: float) -> float:
    return max(abs(rep0-rep1), abs(rep0-rep2), abs(rep1-rep2))


# Returns the median float of three different runtimes
def getMedianRuntime(rep0: float, rep1: float, rep2: float) -> float:
    return sorted([rep0, rep1, rep2])[1]


# Get the consensus answer across triplicates
# If two or more of the triplicate agree, return this majority vote; if all disagree, return randomly just first
def getConsensusJSON(rep0: str, rep1: str, rep2: str) -> str:
    rep0 = json.loads(rep0)
    rep1 = json.loads(rep1)
    rep2 = json.loads(rep2)
    if rep0 == rep1:
        return rep0
    elif rep0 == rep2:
        return rep0
    elif rep1 == rep2:
        return rep1
    else:
        return rep0


# For testing whether a character string can be parsed into proper JSON
def getJSONParseability(text: str) -> bool:
    success = False
    try:
        json.loads(text)
        success = True
    except ValueError:
        success = False
    return success

# Test if all in the triplicates can be parsed into proper JSON
def getJSONParseAll(rep0: str, rep1: str, rep2: str):
    if (not getJSONParseability(rep0)) or (not getJSONParseability(rep1)) or (not getJSONParseability(rep2)):
        return False
    else:
        return True

## Parameters and settings for running the collection

# Quite standard level of terminal output
verbose = False
# Whether to print additional debug info when collecting results
debug = False

# Names of explored models
modelnames = [
    # Claudes (Anthropic AI)
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    # Geminis (Google)
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    #"gemini-1.5-pro-001",
    #"gemini-1.5-pro-002",
    # GPTs (OpenAI)
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-2025-04-14",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    # Grok (xAI)
    "grok-3-beta",
    "grok-2-1212",
    # Replicate runs
    # Llamas
    "meta-llama-4-maverick-instruct",
    "meta-llama-4-scout-instruct",
    # DeepSeek
    "deepseek-ai-deepseek-r1",
    "deepseek-ai-deepseek-v3",
    # Gemmas (non-local deployment)
    "google-deepmind-gemma-3-4b-it",
    "google-deepmind-gemma-3-12b-it",
    "google-deepmind-gemma-3-27b-it",
    # Mistral
    "mistral-large-2411",
    "mistral-medium-2505",
    "mistral-small-2503",
    # Local models run on a RTX 5090 setup
    # Gemmas
    # Comment out; run on Replicate
    #"google/gemma-3-4b-it",
    #"google/gemma-3-12b-it",
    #"google/gemma-3-27b-it",
    # MedGemmas (fine-tuned Gemmas that focus on biomedical data)
    "google/medgemma-4b-it",
    "google/medgemma-27b-it",
    # GPT-OSSes (OpenAI)
    "openai/gpt-oss-20b",
    # Qwens (Alibaba)
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
]
# Model name sanitization to avoid potentially difficult characters
modelnames = list(map(lambda x: re.sub("/", "-", x), modelnames))

# Binary full output save
full = np.load("npydata\\full.npy")
# Binary output save
output = np.load("npydata\\output.npy")
# Binary runtimes save
runtimes = np.load("npydata\\runtimes.npy")
# Example JSON output
print(output[0,0,0,0,0,0,0].decode("utf-8"))
# Example runtimes
print(runtimes[0,0,0,0,0,0,0])

summary = pd.DataFrame({'censoption': [], 'model': [], 'promptIndex': [], 'inputIndex': [], 'lang': [],
                        'seedoption': [],
                        'concordanceExact': [], 'concordanceCaseinsensitive': [], 'concordanceContent': [],
                        'medianCharCount': [], 'maxCharDiff': [], 'maxRuntimeDiff': [],
                        'medianRuntime': [],
                        # Triplicate level
                        'parseable1': [], 'parseable2': [], 'parseable3': [],
                        # Answer as str
                        'answer1rep0': [], 'answer2rep0': [], 'answer3rep0': [],
                        'answer1rep1': [], 'answer2rep1': [], 'answer3rep1': [],
                        'answer1rep2': [], 'answer2rep2': [], 'answer3rep2': [],
                        # Correctness as bool
                        'correct1rep0': [], 'correct2rep0': [], 'correct3rep0': [],
                        'correct1rep1': [], 'correct2rep1': [], 'correct3rep1': [],
                        'correct1rep2': [], 'correct2rep2': [], 'correct3rep2': [],
                        # Consensus level
                        'parseable': [],
                        'answer1': [], 'answer2': [], 'answer3': [], #'consensusAnswer': [],
                        'correct1': [], 'correct2': [], 'correct3': [], 'allCorrect': []
                        })


# Dimensions in order: replicates (3), models, prompts, inputs, languages (3 with gaps)
print("\n\nDimensions and shape of output: \n")
print(str(output.ndim) + "\n")
print(str(output.shape) + "\n")
print("\n\nDimensions and shape of runtimes: \n")
print(str(runtimes.ndim) + "\n")
print(str(runtimes.shape) + "\n")

for cens in [0, 1]:
    for modelIndex in range(len(modelnames)):
        for promptIndex in range(histoPCaData.getMaxPromptLength()):
            for lang in [0, 1]:
                for inputIndex in range(histoPCaData.getMaxInputLength(lang)):
                    # Select seed variants based on if it was allowed (GPT and Llama families)
                    #if bool(re.search("gpt|llama", modelnames[modelIndex], re.IGNORECASE)):
                    #    seeds = range(2)
                    # Other families are run just with "seedFalse" results (seed = False)
                    #else:
                    #    seeds = range(1)
                    #for seedoption in seeds:
                    # Only handle seed False for now
                    for seedoption in [0]:
                        if debug:
                            print("\n\n--- Triplicates (full output, parsed output, runtimes) ---")
                            print("Model : " + modelnames[modelIndex] + "\n\n")
                            print(full[cens, seedoption, 0, modelIndex, promptIndex, inputIndex, lang].decode("utf-8"))
                            print(output[cens, seedoption, 0, modelIndex, promptIndex, inputIndex, lang].decode("utf-8"))
                            print(runtimes[cens, seedoption, 0, modelIndex, promptIndex, inputIndex, lang])
                            print("\n")
                            print(full[cens, seedoption, 1, modelIndex, promptIndex, inputIndex, lang].decode("utf-8"))
                            print(output[cens, seedoption, 1, modelIndex, promptIndex, inputIndex, lang].decode("utf-8"))
                            print(runtimes[cens, seedoption, 1, modelIndex, promptIndex, inputIndex, lang])
                            print("\n")
                            print(full[cens, seedoption, 2, modelIndex, promptIndex, inputIndex, lang].decode("utf-8"))
                            print(output[cens, seedoption, 2, modelIndex, promptIndex, inputIndex, lang].decode("utf-8"))
                            print(runtimes[cens, seedoption, 2, modelIndex, promptIndex, inputIndex, lang])
                            print("\n")
                            time.sleep(1)

                        try:
                            full1 = full[cens, seedoption, 0, modelIndex, promptIndex, inputIndex, lang].decode("utf-8")
                            full2 = full[cens, seedoption, 1, modelIndex, promptIndex, inputIndex, lang].decode("utf-8")
                            full3 = full[cens, seedoption, 2, modelIndex, promptIndex, inputIndex, lang].decode("utf-8")
                            output1 = output[cens, seedoption, 0, modelIndex, promptIndex, inputIndex, lang].decode("utf-8")
                            output2 = output[cens, seedoption, 1, modelIndex, promptIndex, inputIndex, lang].decode("utf-8")
                            output3 = output[cens, seedoption, 2, modelIndex, promptIndex, inputIndex, lang].decode("utf-8")
                            runtime1 = runtimes[cens, seedoption, 0, modelIndex, promptIndex, inputIndex, lang]
                            runtime2 = runtimes[cens, seedoption, 1, modelIndex, promptIndex, inputIndex, lang]
                            runtime3 = runtimes[cens, seedoption, 2, modelIndex, promptIndex, inputIndex, lang]
                            parseable = (getJSONParseability(output1) and getJSONParseability(output2) and
                                         getJSONParseability(output3))
                        except ValueError as e:
                            print("Error processing at:\ncens" + str(cens) + "\nseedoption" + str(seedoption) +
                                  "\nmodelIndex" + str(modelIndex) + "\nmodelname" + modelnames[modelIndex] +
                                  "\npromptIndex" + str(promptIndex) +
                                  "\ninputIndex" + str(inputIndex) + "\nlang" + str(lang))
                            parseable = False

                        answer1 = "NA"
                        answer2 = "NA"
                        answer3 = "NA"
                        if parseable:
                            dat = getConsensusJSON(output1, output2, output3)
                            if dat.__class__ == dict:
                                vals = list(dat.values())
                                if len(vals) == 3:
                                    answer1 = str(vals[0])
                                    answer2 = str(vals[1])
                                    answer3 = str(vals[2])
                                else:
                                    answer1 = "wrong length"
                                    answer2 = "wrong length"
                                    answer3 = "wrong length"
                            else:
                                answer1 = "wrong class"
                                answer2 = "wrong class"
                                answer3 = "wrong class"
                        else:
                            answer1 = "cannot parse"
                            answer2 = "cannot parse"
                            answer3 = "cannot parse"

                        # Correct answer to the presented questions, regular expression format
                        correct1 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 0)), str(answer1), re.IGNORECASE))
                        correct2 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 1)), str(answer2), re.IGNORECASE))
                        correct3 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 2)), str(answer3), re.IGNORECASE))
                        # Extract individual answers and use RegEx to check against the correct answer
                        answer1rep0 = answer1rep1 = answer1rep2 = "<NA>"
                        answer2rep0 = answer2rep1 = answer2rep2 = "<NA>"
                        answer3rep0 = answer3rep1 = answer3rep2 = "<NA>"
                        correct1rep0 = correct1rep1 = correct1rep2 = "<NA>"
                        correct2rep0 = correct2rep1 = correct2rep2 = "<NA>"
                        correct3rep0 = correct3rep1 = correct3rep2 = "<NA>"
                        dat = vals = "<NA>"
                        if getJSONParseability(output1):
                            dat = json.loads(output1)
                            vals = list(dat.values())
                            answer1rep0 = str(vals[0])
                            answer2rep0 = str(vals[1])
                            answer3rep0 = str(vals[2])
                            correct1rep0 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 0)), answer1rep0, re.IGNORECASE))
                            correct2rep0 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 1)), answer2rep0, re.IGNORECASE))
                            correct3rep0 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 2)), answer3rep0, re.IGNORECASE))
                        if getJSONParseability(output2):
                            dat = json.loads(output2)
                            vals = list(dat.values())
                            answer1rep1 = str(vals[0])
                            answer2rep1 = str(vals[1])
                            answer3rep1 = str(vals[2])
                            correct1rep1 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 0)), answer1rep1, re.IGNORECASE))
                            correct2rep1 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 1)), answer2rep1, re.IGNORECASE))
                            correct3rep1 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 2)), answer3rep1, re.IGNORECASE))
                        if getJSONParseability(output3):
                            dat = json.loads(output3)
                            vals = list(dat.values())
                            answer1rep2 = str(vals[0])
                            answer2rep2 = str(vals[1])
                            answer3rep2 = str(vals[2])
                            correct1rep2 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 0)), answer1rep2, re.IGNORECASE))
                            correct2rep2 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 1)), answer2rep2, re.IGNORECASE))
                            correct3rep2 = bool(re.search(str(histoPCaData.getInputAnswer(inputIndex, cens, 2)), answer3rep2, re.IGNORECASE))

                        # Append the outcome from the triplicates
                        summary.loc[len(summary.index)] = [
                            cens,
                            modelnames[modelIndex],
                            promptIndex,
                            inputIndex,
                            lang,
                            str([False, True][seedoption]),
                            getConcordanceExact(full1, full2, full3),
                            getConcordanceCaseinsensitive(full1, full2, full3),
                            getConcordanceContent(output1, output2, output3),
                            getMedianCharCount(full1, full2, full3),
                            getMaxCharDiff(full1, full2, full3),
                            getMaxRuntimeDiff(runtime1, runtime2, runtime3),
                            getMedianRuntime(runtime1, runtime2, runtime3),
                            # Reporting all in the triplicate
                            getJSONParseability(output1),
                            getJSONParseability(output2),
                            getJSONParseability(output3),
                            # Answers as str
                            answer1rep0, answer2rep0, answer3rep0,
                            answer1rep1, answer2rep1, answer3rep1,
                            answer1rep2, answer2rep2, answer3rep2,
                            # Correctness as bool
                            correct1rep0, correct2rep0, correct3rep0,
                            correct1rep1, correct2rep1, correct3rep1,
                            correct1rep2, correct2rep2, correct3rep2,
                            # Consensus answers
                            parseable, # If all triplicates could be parsed to JSON correctly
                            answer1, # Consensus answer to the first query ("Sample type")
                            answer2, # Consensus answer to the first query ("Is it malignant")
                            answer3, # Consensus answer to the first query ("If malignant, what is the Gleason")
                            correct1, # If the first question ("Sample type") is correct in consensus JSON
                            correct2, # If the second question ("Is it malignant") is correct in consensus JSON
                            correct3, # If the third question ("If malignant, what is Gleason") is correct in consensus JSON
                            correct1 & correct2 & correct3 # If all 3 queries were correct
                        ]

print("Finished collecting")
print(summary)
summary.to_csv('summary.tsv', sep="\t")
