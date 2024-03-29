from sklearn.metrics import cohen_kappa_score


def get_kappas(df1, df2):
    """
    Get kappa scores for sentiment and emotion annotations
    """
    valid_emotions = [
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutral",
        "sadness",
        "surprise",
    ]
    valid_sentiments = ["negative", "neutral", "positive"]

    df1 = df1[
        df1["sentiment"].isin(valid_sentiments) & df1["emotion"].isin(valid_emotions)
    ]
    df2 = df2[
        df2["sentiment"].isin(valid_sentiments) & df2["emotion"].isin(valid_emotions)
    ]

    sent_kappa = cohen_kappa_score(df1["sentiment"], df2["sentiment"])
    emo_kappa = cohen_kappa_score(df1["emotion"], df2["emotion"])

    print(f"Sentiment Kappa: {sent_kappa}")
    print("Counts for sentiments by df1:")
    print(df1["sentiment"].value_counts())
    print("Counts for sentiments by df2:")
    print(df2["sentiment"].value_counts())
    print(f"Emotion Kappa: {emo_kappa}")
    print("Emotion counts for df1:")
    print(df1["emotion"].value_counts())
    print("Emotion counts for df2:")
    print(df2["emotion"].value_counts())
