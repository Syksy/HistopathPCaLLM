# 102 statements; 3 languages, non-censored vs. censored
statements = [
    # Statement 000
    [
        # Non-censored
        [
            # English (non-censored)
            ["Chipping samples totaled approx 30g, of which 20g were started in six blocks. In histological examination benign prostate tissue was observed, with no atypia, malignancies, or carcinoma. Some spot-like inflammation can be observed. The finding thus fits the suspected benign prostate hyperplasia.", "TURP", "no|ei|nej", "^NA$"]
            # Finnish (non-censored)
            ["Höyläyslastuja tuli yhteensä noin 30g, joista noin 20g käynnistettiin kuudessa blokissa. Histologisessa tarkastelussa havaitaan hyvänlaatuista prostatakudosta, missä ei atypiaa, malignia, tai karsinoomaa näy. Vähän nähtävissä läiskittäistä tulehdusta. Löydös sopii siis epäiltyyn hyvänlaatuiseen hyperplasiaan.", "TURP", "no|ei|nej", "^NA$"],
            # Swedish (non-censored)
            ["", "TURP", "no|ei|nej", "^NA$"]
        ],
        # Censored
        [
            # English (censored)
            ["A lot sample material was obtained, which were run in multiple batches. In histological examination prostate tissue was observed, but any inference of cell atypia, malignancy, or carcinoma cannot be conducted. Some spot-like inflammation can be observed. From this material reliable conclusions cannot be drawn regarding whether the findings fit the suspicion of benign prostate hyperplasia.", "^NA$", "^NA$", "^NA$"],
            # Finnish (censored)
            ["Näytemateriaalia tuli yhteensä runsaasti, josta suurin osa käynnistettiin useammassa erässä. Histologisessa tarkastelussa havaitaan prostatakudosta, mutta päättelyä solujen atypiasta, maligniteetista, tai karsinoomasta ei voida tehdä. Vähän nähtävissä läiskittäistä tulehdusta. Materiaalista ei voida vetää johtopäätöstä sopisiko se pelkästään epäiltyyn hyvänlaatuiseen hyperplasiaan.", "^NA$", "^NA$", "^NA$"],
            # Swedish (censored)
            ["", "^NA$", "^NA$", "^NA$"]
        ]
    ],
    # Statement 001
    [
        # Non-censored
        [
            # English (non-censored)
            ["", "", "", ""],
            # Finnish (non-censored)
            ["", "", "", ""],
            # Swedish (non-censored)
            ["", "", "", ""]
        ],
        # Censored
        [
            # English (censored)
            ["", "^NA$", "^NA$", "^NA$"],
            # Finnish (censored)
            ["", "^NA$", "^NA$", "^NA$"],
            # Swedish (censored)
            ["", "^NA$", "^NA$", "^NA$"]
        ]
    ],

    # ...
    # Statement 102
    [
        # Non-censored
        [
            # English (non-censored)
            ["Transurethral resection chippings were obtained after operation, totalling 10g in sample weight. Examination of the chippings provided evidence of adenocarcinoma with heterogeneous growth patterns, with Gleason grading major 4 and minor 3. The identified minor components were distributed focally, totalling around 10% of tumor area, with the major component prominent in whole sample (totalling around 40% of tumor area). The non-tumorous area exhibited benign hyperplasia, with further chronic inflammation patterns. The ISUP Grade Group 3 suggests that patient should be possibly screened or biopsied further in addition to his on-going active surveillance, as cribriform-like patterns were pointed out by colleague John Doe, MD, during post-BPH treatment consultation.", "TURP", "yes|kyllä|ja", "^4\\+3=7$"],
            # Finnish (non-censored)
            ["", "TURP", "yes|kyllä|ja", "^4\\+3=7$"],
            # Swedish (non-censored)
            ["", "TURP", "yes|kyllä|ja", "^4\\+3=7$"]
        ],
            # Censored
        [
            # English (censored)
            ["", "^NA$", "^NA$", "^NA$"],
            # Finnish (censored)
            ["", "^NA$", "^NA$", "^NA$"],
            # Swedish (censored)
            ["", "^NA$", "^NA$", "^NA$"]
        ]
    ]
]
