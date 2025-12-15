import pandas as pd


def get_top_features(vectorizer, model, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    top_positive = sorted(
        zip(feature_names, coefficients),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    top_negative = sorted(
        zip(feature_names, coefficients),
        key=lambda x: x[1]
    )[:top_n]

    pos_df = pd.DataFrame(top_positive, columns=["Word", "Weight"])
    neg_df = pd.DataFrame(top_negative, columns=["Word", "Weight"])

    return pos_df, neg_df
