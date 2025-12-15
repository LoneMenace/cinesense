# 🎬 CineSense – Movie Review Sentiment Analysis App

CineSense is a web application that analyzes movie reviews to determine sentiment and confidence,
while also exposing the exact numerical reasoning behind each prediction.

The application is containerized using Docker and deployed on an AWS EC2 instance.

---

## ✨ Features

- 🧪 **Sentence-level sentiment analysis** (Positive / Negative)
- 📊 **Confidence scoring** based on model probability
- 🔍 **Word-level explainability**
  - Recognized words from the trained vocabulary
  - Learned weight contribution of each word
- 🧮 **Numerical decision breakdown**
  - Sum of word weights
  - Model intercept
  - Sigmoid probability calculation
- 💾 **Persistent storage** using SQLite
- 🗑️ **Per-review deletion** directly from the UI
- 📈 **Global model insights**
  - Strongest positive and negative indicators learned during training

---

## 🧠 How Sentiment Is Computed

1. Input text is vectorized using the trained IMDb vocabulary  
2. Only words present in the training vocabulary are retained  
3. Each retained word contributes a learned numerical weight  
4. Word weights are summed with the model intercept  
5. The final score determines sentiment polarity  

- Positive score → **Positive sentiment**  
- Negative score → **Negative sentiment**

---

## 📊 How Confidence Is Calculated

The final score is converted into a probability using a sigmoid function:

Probability = 1 / (1 + e^(-score))


- If the prediction is **Positive**, confidence = positive probability  
- If the prediction is **Negative**, confidence = 1 − positive probability  

Stronger sentiment signals produce probabilities closer to 0% or 100%, resulting in higher confidence.

---

## 🛠 Tech Stack

**Application**
- Python
- Streamlit
- scikit-learn

**Data**
- SQLite (persistent local storage)

**Cloud & Deployment**
- Docker (containerized runtime)
- AWS EC2 (hosting and execution)

**Version Control**
- Git & GitHub

---

## ☁️ Deployment Overview

- The application runs inside a Docker container
- The container is deployed on an AWS EC2 instance
- The app is accessed via the EC2 public IP and exposed port
- SQLite data persists across container and instance restarts

This setup ensures consistent runtime behavior and reproducible deployments.

---

## ⚠️ Model Constraints

- The model is trained only on **English-language IMDb reviews**
- Words not present in the training vocabulary are ignored
- Mixed or neutral input may result in lower confidence scores

These behaviors are visible in the application output.

---

## 📄 License

MIT License
