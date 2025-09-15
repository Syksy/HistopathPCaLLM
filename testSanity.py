# Sanity checking and tests of prompts, input synthetic statements etc
import os
import time
import re
import pandas as pd
# Running models on Replicate
import replicate
# Import synthetic data and prompts
import histoPCaData

# API key load, root directory etc
from dotenv import load_dotenv
load_dotenv()

# Working directory for project taken from env vars
os.chdir(os.path.join(os.environ.get("ROOT_DIR"), "sanity"))

# Prompt to use as sanity checking whether there's signal or not
prompt = ("You should act as a prostate cancer histopathologist, who will examine validity of the provided statements. You will be provided by a prostate cancer histopathology statement, which is in English, Finnish, or Swedish. You should assess whether the statement is informative or not; if it is informative, it contains information of what is the sample type (such as biopsies, turp, or radical prostatectomy), and whether there were malignant findings (e.g. prostate adenocarcinoma) or just verified benign findings (note that discrepancies and inconclusiveness do not indicate benign status of the examination). If the statement indicates that the finding is malignant, it should also contain the information for Gleason, such as major, minor and sum scores, or a valid way of inferring these.\nYour answer should contain the following six sections, preceded with the characters # and number, where the parts between < and > should be filled:\n"
          "#1: <Whether sample type can be assessed; valid options are Yes, No, Partially, Not applicable>\n"
          "#2: <Verbal explanation to your assessment of section #1>\n"
          "#3: <Whether malignancy of findings can be assessed: valid options are Yes, No, Partially, Not applicable>\n"
          "#4: <Verbal explanation to your assessment of section #3>\n"
          "#5: <Whether Gleason in format major+minor=sum can be assessed: valid options are Yes, No, Partially, Not applicable>\n"
          "#6: <Verbal explanation to your assessment of section #5>\n"
          "#7: <Whether the grammar and level of understanding are appropriate to e.g. a layman, but do notice that some non-standard grammar may be used in the domain: valid options are Appropriate, Minor mistakes, Major mistakes, Incomprehensible>"
          "#8: <Verbal explanation to your assessment of section #7, which may include correction suggestions>"
          "\n\nAssess the following statement and give your answers primarily in English, regardless of the original language:\n\n\'")
# Use an assisting system prompt
system_prompt = "You are an expert histopathologist, who will critically review provided statements on their informativeness. Pay attention to returning the answers in the specified format to the user. Avoid unnecessary verbosity or extra line changes."

# Iterate across the models and combinations
while True:
    try:
        # Input statements
        for inputIndex in histoPCaData.getArrayInputIndex():
            # Iterate across languages (0 = English, 1 = Finnish, 2 = Swedish)
            for lang in [0, 1, 2]:
                # Non-censored (value 0) or censored (1) version of the input statements
                for cens in [0, 1]:
                    # Iterate across models to use for sanity checking
                    for modelname in [
                        "moonshotai/kimi-k2-instruct",
                        "openai/o1",
                    ]:
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
                            print("\nRun for file " + filename + "\n\n")
                            out = ""
                            # Instructions provided in the prompt followed by the statement to be sanity checked
                            query = prompt + statement + '\''
                            print("\nQuery going in:\n\n" + query)
                            f = open(filename + ".check", 'w', encoding="utf-8")
                            for iterator in replicate.run(
                                    modelname,
                                    input={
                                        "prompt": query,
                                        "system_prompt": system_prompt
                                    }
                            ):
                                out = out + "".join(iterator)
                                print("".join(iterator), end="", file=f)
                            print("\nOutput:\n\n" + out)
        break
    except Exception as e:
        print("Error: " + str(e))
        time.sleep(120)

sanity = pd.DataFrame({'model': [],
                        'inputIndex': [],
                        'lang': [],
                        'cens': [],
                        'hit1': [], # Can sample type be inferred
                        'hit2': [], # Verbal explanation
                        'hit3': [], # Can malignancy be inferred
                        'hit4': [], # Verbal explanation
                        'hit5': [], # Can Gleason be inferred
                        'hit6': [], # Verbal explanation
                        'hit7': [], # Grammar and clarity category
                        'hit8': []  # Verbal explanation
                        })

# Collect results into a pandas, and pop out a tsv
# Iterate across models to use for sanity checking
for modelname in [
    "moonshotai/kimi-k2-instruct",
    "openai/o1",
]:
    # Input statements
    for inputIndex in histoPCaData.getArrayInputIndex():
        # Iterate across languages (0 = English, 1 = Finnish, 2 = Swedish)
        for lang in [0, 1, 2]:
            # Non-censored (value 0) or censored (1) version of the input statements
            for cens in [0, 1]:
                # Construct filename
                filename = ("SanityCheck_" + re.sub("/", "-", modelname)
                            + "_input" + str(inputIndex)
                            + "_lang" + str(lang)
                            + "_cens" + str(cens)
                            )
                # Construct sanity check and add to pandas
                if os.path.isfile(filename + ".check"):
                    file = open(filename + ".check", 'r', encoding="utf-8")
                    lines = file.readlines()
                    lines = "".join(lines).strip()
                    # Parameter combo and full output prior to any potential parsing;
                    # going from {"#1", "#2", ..., "#8"} line starters
                    sanity.loc[len(sanity.index)] = [
                        modelname,
                        inputIndex,
                        lang,
                        cens,
                        # Remove extra angle brackets after finding the answers preceded by #num
                        re.sub(r"[<>]", "", "".join(re.findall(r"^#1\s*(.*)$", lines, flags=re.MULTILINE))),
                        re.sub(r"[<>]", "", "".join(re.findall(r"^#2\s*(.*)$", lines, flags=re.MULTILINE))),
                        re.sub(r"[<>]", "", "".join(re.findall(r"^#3\s*(.*)$", lines, flags=re.MULTILINE))),
                        re.sub(r"[<>]", "", "".join(re.findall(r"^#4\s*(.*)$", lines, flags=re.MULTILINE))),
                        re.sub(r"[<>]", "", "".join(re.findall(r"^#5\s*(.*)$", lines, flags=re.MULTILINE))),
                        re.sub(r"[<>]", "", "".join(re.findall(r"^#6\s*(.*)$", lines, flags=re.MULTILINE))),
                        re.sub(r"[<>]", "", "".join(re.findall(r"^#7\s*(.*)$", lines, flags=re.MULTILINE))),
                        re.sub(r"[<>]", "", "".join(re.findall(r"^#8\s*(.*)$", lines, flags=re.MULTILINE)))
                    ]

# Working directory for project taken from env vars
os.chdir(os.environ.get("ROOT_DIR"))

print(sanity)
sanity.to_csv('sanity.tsv', sep="\t")
