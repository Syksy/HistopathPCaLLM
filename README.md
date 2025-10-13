# HistopathPCaLLM

This repository contains evaluation of Large Language Models' (LLMs') ability 
to structure three fields queried from synthetic histopathological prostate 
cancer samples
1. What is the source / method of the sample (biopsy, RP, or TURP)
2. Where there reported malignant findings (yes or no)
3. If there were malignant findings, what was the found Gleason grade (major+minor=sum)

Three languages were tested: English, Finnish, and Swedish.
Manual human translations were done across the three languages.
In addition to the statements containing reasonable signal to answer the questions,
censored versions were human-curated which resembled real statements in structure, but
for various reasons could not be used to answer any of the three questions,
thus they should return 'NA' values by the LLMs.

# Files

The repository contains both the synthetic data files as well as the run files 
intended for calling LLMs via an API or locally. The files are as follows:

- histoPCaData.py : Contains the base synthetic statements in three languages as well as the utilized prompts
- histoPCaCollect.py : 
- histoPCaTabulation.py :

- runClaude.py :
- runGemini.py :
- runGPT.py :
- runGrok.py :
- runLocal.py :
- runMistral.py :
- runReplicate.py :

- testSanity.py :

# Folders

In addition to files in the root folder, key folders contain:

- /out/ : Contains the output provided by all models
- /sanity/ : Sanity check output for statements (by GPT o1 and Kimi K2)

The output file suffixes are as follows:

- *.out : The raw output as text
- *.prompt : The original prompt (containing the statement) that was sent to the LLM
- *.time : Wall clock time from starting the query to obtaining a response from the LLM

# Local setup

TBA

# Prompt(s)

TBA

# Data

TBA

# Citation

TBA

# Contact

For email queries in addition to GitHub: daniel.laajala@helsinki.fi / teelaa@utu.fi
