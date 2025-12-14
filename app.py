import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ================= LOAD MODEL =================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="CineSense | Movie Review Sentiment Analyzer",
    layout="centered"
)

# ================= HEADER =================
st.markdown(
    """
    <h1 style='text-align: center;'>🎬 CineSense</h1>
    <h4 style='text-align: center; color: gray;'>
    Movie Review Sentiment Analyzer
    </h4>
    <br>
    """,
    unsafe_allow_html=True
)

st.write(
    "Analyze the sentiment of movie reviews using a machine learning model trained on the IMDb dataset. "
    "Enter one or multiple sentences below to get detailed insights."
)

st.divider()

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= INPUT SECTION =================
st.subheader("✍️ Review Input")

text = st.text_area(
    "Enter your review text (one sentence per line)",
    height=170,
    placeholder=(
        "I went in with low expectations, but the movie surprised me.\n"
        "The acting was phenomenal.\n"
        "The plot felt weak in parts."
    )
)

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    clear = st.button("🧹 Clear Session History", use_container_width=True)

if clear:
    st.session_state.history = []
    st.success("Session history cleared.")

# ================= ANALYSIS =================
if analyze:
    if text.strip() == "":
        st.warning("Please enter at least one sentence to analyze.")
    else:
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        X = vectorizer.transform(sentences)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        for sent, pred, prob in zip(sentences, predictions, probabilities):
            confidence = prob[1] if pred == 1 else prob[0]

            st.session_state.history.append({
                "Sentence": sent,
                "Sentiment": "Positive" if pred == 1 else "Negative",
                "Confidence (%)": round(confidence * 100, 2)
            })

        st.success("Sentiment analysis completed successfully.")

# ================= RESULTS =================
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    st.divider()
    st.subheader("📊 Analysis Results")

    # ---- Sentence-level table ----
    st.markdown("**Sentence-level Predictions**")
    st.dataframe(df, use_container_width=True)

    # ---- Overall sentiment ----
    pos_count = (df["Sentiment"] == "Positive").sum()
    neg_count = (df["Sentiment"] == "Negative").sum()

    st.subheader("🧠 Overall Review Sentiment")

    if pos_count >= neg_count:
        st.success("Overall Sentiment: **Positive** 😊")
    else:
        st.error("Overall Sentiment: **Negative** 😞")

    # ---- Average confidence ----
    avg_conf = df["Confidence (%)"].mean()
    st.subheader("📈 Average Confidence Level")
    st.progress(int(avg_conf))
    st.write(f"Average confidence across sentences: **{avg_conf:.2f}%**")

    # ---- Distribution chart ----
    st.subheader("📉 Sentiment Distribution")

    fig, ax = plt.subplots()
    ax.bar(["Positive", "Negative"], [pos_count, neg_count])
    ax.set_ylabel("Number of Sentences")
    ax.set_title("Positive vs Negative Sentences")
    st.pyplot(fig)

    # ---- Explainability ----
    st.subheader("🔍 Model Explainability")

    st.write(
        "The model determines sentiment based on learned word importance from IMDb reviews. "
        "Below are the strongest indicators influencing predictions."
    )

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    top_positive = sorted(
        zip(feature_names, coefficients),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    top_negative = sorted(
        zip(feature_names, coefficients),
        key=lambda x: x[1]
    )[:10]

    colp, coln = st.columns(2)

    with colp:
        st.markdown("**🟢 Strong Positive Indicators**")
        st.table(pd.DataFrame(top_positive, columns=["Word", "Weight"]))

    with coln:
        st.markdown("**🔴 Strong Negative Indicators**")
        st.table(pd.DataFrame(top_negative, columns=["Word", "Weight"]))

# ================= FOOTER =================
st.divider()
st.caption(
    "CineSense is a demo project built using Python, Streamlit, and scikit-learn. "
    "Designed for cloud deployment and DevOps workflows."
)
