import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
import math

from database import (
    create_table,
    insert_review,
    fetch_recent_reviews,
    delete_review
)
from explain import get_top_features

# ================= INIT =================
create_table()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
intercept = model.intercept_[0]

st.set_page_config(page_title="CineSense", layout="centered")

# ================= HEADER =================
st.markdown(
    """
    <h1 style="text-align:center;">🎬 CineSense</h1>
    <h4 style="text-align:center; color:#6c757d;">
    Sentiment Analysis for Movie Reviews
    </h4>
    <p style="text-align:center; max-width:720px; margin:auto;">
    Analyze movie reviews to determine sentiment
    and the exact numerical reasoning behind each prediction.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ================= TABS =================
tab_analyze, tab_history, tab_about = st.tabs(
    ["🧪 Analyze", "📜 Past Reviews", "ℹ️ About"]
)

# =====================================================
# 🧪 ANALYZE TAB
# =====================================================
with tab_analyze:
    st.subheader("✍️ Analyze Reviews")

    st.caption(
    "⚠️ Note: This model is trained only on English reviews. "
    "Non-English words or sentences may be ignored or lead to unreliable predictions."
     )


    text = st.text_area(
        "Enter one or more sentences (one per line)",
        height=160,
        placeholder=(
            "The performances were outstanding.\n"
            "The story lost momentum halfway through."
        )
    )

    analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("Please enter at least one sentence.")
        else:
            sentences = [s.strip() for s in text.split("\n") if s.strip()]
            X = vectorizer.transform(sentences)
            preds = model.predict(X)

            for sent, pred in zip(sentences, preds):
                # ---------- VECTOR & WORD EXTRACTION ----------
                sent_vec = vectorizer.transform([sent]).toarray()[0]
                active_indices = sent_vec.nonzero()[0]

                word_details = [
                    (feature_names[i], coefficients[i])
                    for i in active_indices
                ]

                weights = [w[1] for w in word_details]

                # ---------- SCORE CALCULATION ----------
                word_sum = sum(weights)
                final_score = word_sum + intercept

                exp_val = math.exp(-final_score)
                prob_positive = 1 / (1 + exp_val)
                prob_negative = 1 - prob_positive

                sentiment = "Positive" if pred == 1 else "Negative"
                confidence = prob_positive if pred == 1 else prob_negative
                confidence_pct = round(confidence * 100, 2)

                insert_review(sent, sentiment, confidence_pct)

                # ---------- RESULT CARD ----------
                color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
                emoji = "🟢" if sentiment == "Positive" else "🔴"

                st.markdown(
                    f"""
                    <div style="
                        border-left:6px solid {color};
                        padding:14px;
                        background-color:#1f1f1f;
                        color:#eaeaea;
                        border-radius:6px;
                        margin-bottom:14px;
                    ">
                        <strong>{emoji} {sentiment}</strong><br>
                        {sent}<br>
                        <small>Confidence: {confidence_pct}%</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ---------- WORD WEIGHTS ----------
                st.markdown("**Recognized words and learned weights:**")

                calc_df = pd.DataFrame(
                    word_details, columns=["Word", "Weight"]
                ).sort_values("Weight", ascending=False)

                st.dataframe(
                    calc_df.style.set_table_styles([
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#343a40"),
                                ("color", "white")
                            ]
                        }
                    ]),
                    use_container_width=True
                )

                # ---------- CONTRIBUTING WORDS SUMMARY ----------
                positive_words = [w for w, wt in word_details if wt > 0]
                negative_words = [w for w, wt in word_details if wt < 0]

                st.markdown(
                    f"""
                    <div style="
                     display:flex;
                     gap:16px;
                     margin-top:10px;
                     margin-bottom:6px;
                    ">
                    <div style="
                     flex:1;
                     padding:12px;
                     background:rgba(46, 204, 113, 0.12);
                     border-left:4px solid #2ecc71;
                     border-radius:6px;
                    ">
                  <strong style="color:#2ecc71;">🟢 Words pushing positive sentiment</strong><br>
                  <span style="color:#eaeaea;">
                  {", ".join(positive_words) if positive_words else "None"}
                  </span>
                     </div>

                     <div style="
                       flex:1;
                       padding:12px;
                       background:rgba(231, 76, 60, 0.12);
                       border-left:4px solid #e74c3c;
                       border-radius:6px;
                     ">
                     <strong style="color:#e74c3c;">🔴 Words pushing negative sentiment</strong><br>
                     <span style="color:#eaeaea;">
                     {", ".join(negative_words) if negative_words else "None"}
                  </span>
                      </div>
                      </div>
                       """,
                       unsafe_allow_html=True
                   )



                # ---------- NUMERICAL DECISION BREAKDOWN ----------
                weight_expansion = " + ".join([f"{w:.2f}" for w in weights])

                st.markdown(
                    f"""
                    <div style="
                        background-color:#f8f9fa;
                        padding:14px;
                        border-radius:6px;
                        color:#212529;
                        margin-top:10px;
                    ">
                    <strong>Numerical decision breakdown:</strong><br><br>

                    Σ Word weights = {weight_expansion}
                    = <strong>{word_sum:.2f}</strong><br>

                    Intercept (model bias) = <strong>{intercept:.2f}</strong><br>
                    <hr style="margin:6px 0;">

                    Final score = <strong>{final_score:.2f}</strong><br><br>

                    Sigmoid(score) = 
                    <code>1 / (1 + e<sup>-score</sup>)</code><br><br>

                    Sigmoid({final_score:.2f}) =
                    <code>1 / (1 + e<sup>-{final_score:.2f}</sup>)</code><br>
                    e<sup>-{final_score:.2f}</sup> = <strong>{exp_val:.6f}</strong><br><br>

                    Positive probability<br>
                    = 1 / (1 + {exp_val:.6f})<br>
                    = <strong>{prob_positive:.6f}</strong><br><br>

                    Negative probability<br>
                    = 1 − {prob_positive:.6f}<br>
                    = <strong>{prob_negative:.6f}</strong><br><br>

                    The higher probability determines the final sentiment.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.divider()


            st.markdown("### 🔍 Global Model Explanation")

            st.write(
                "These tables show the strongest keywords learned during training "
                "and how they influence sentiment across many reviews."
            )

            pos_df, neg_df = get_top_features(vectorizer, model)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🟢 **Strong Positive Indicators**")
                st.table(pos_df)

            with col2:
                st.markdown("🔴 **Strong Negative Indicators**")
                st.table(neg_df)

# =====================================================
# 📜 PAST REVIEWS TAB
# =====================================================
with tab_history:
    st.subheader("📜 Past Reviews")

    reviews = fetch_recent_reviews(limit=30)

    if not reviews:
        st.info("No reviews stored yet.")
    else:
        for review_id, review, sentiment, confidence, timestamp in reviews:
            color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
            emoji = "🟢" if sentiment == "Positive" else "🔴"

            time_fmt = datetime.strptime(
                timestamp, "%Y-%m-%d %H:%M:%S"
            ).strftime("%d %b %Y, %H:%M")

            col1, col2 = st.columns([0.9, 0.1])

            with col1:
                st.markdown(
                    f"""
                    <div style="
                        border-left:6px solid {color};
                        padding:14px;
                        background-color:#1f1f1f;
                        color:#eaeaea;
                        border-radius:6px;
                        margin-bottom:10px;
                    ">
                        <strong>{emoji} {sentiment}</strong>
                        <span style="color:#b0b0b0;"> — {time_fmt}</span><br>
                        {review}<br>
                        <small>Confidence: {confidence:.2f}%</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                if st.button("🗑️", key=f"del_{review_id}"):
                    delete_review(review_id)
                    st.rerun()


# =====================================================
# ℹ️ ABOUT TAB
# =====================================================
with tab_about:
    st.markdown(
        """
        ### ℹ️ About CineSense

        CineSense is a movie review sentiment analysis web application that allows users
        to submit text reviews and receive sentiment predictions with confidence scores
        and detailed, data-backed explanations.

        The application stores analyzed reviews using SQLite, enabling persistence across
        restarts and deployments.

        CineSense is deployed on an AWS EC2 instance and runs inside a Docker container.
        Containerization ensures consistent runtime behavior, simplified dependency
        management, and repeatable deployments across environments.

        **Key Components**
        - Streamlit-based web interface  
        - Trained scikit-learn NLP model  
        - SQLite database for persistent storage  
        - Dockerized application runtime  
        - AWS EC2 for hosting and execution  
        """
    )

# ================= FOOTER =================
st.divider()
st.markdown(
    """
    <p style="text-align:center; color:#6c757d; font-size:0.85em;">
    CineSense · Containerized Sentiment Analysis App · Hosted on AWS EC2
    </p>
    """,
    unsafe_allow_html=True
)
