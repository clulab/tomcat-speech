def genderclassifier(f0_list):
    n = len(f0_list)
    beg = int(round(0.25 * n))
    end = int(round(0.75 * n))
    f0 = pd.DataFrame(f0_list)

    df = f0.iloc[
        beg:end,
    ]
    val = df[(df["F0final_sma"] >= 50) & (df["F0final_sma"] <= 360)]
    f0_val = mean(val["F0final_sma"])

    model = pd.read_csv("Downloads/datasets/f0_list.csv")
    f0_m = model[model["gender"] == 2].mean()
    f0_f = model[model["gender"] == 1].mean()

    diff1 = f0_val - f0_m["f0"]
    diff2 = f0_val - f0_f["f0"]
    gender_unknown = None
    if diff1 <= 0:
        gender_unknown = 2
    elif diff2 >= 0:
        gender_unknown = 1
    elif abs(diff2) > abs(diff1):
        gender_unknown = 2
    elif abs(diff1) >= abs(diff2):
        gender_unknown = 1
    return gender_unknown
