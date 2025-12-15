# 🎬 CineSense – Movie Review Sentiment Analysis App

**CineSense** is a cloud-deployed web application that analyzes movie reviews to determine sentiment
with full numerical transparency. Unlike typical sentiment analyzers, CineSense exposes the
**exact words, learned weights, and mathematical calculations** used by the model for every prediction.

The application is containerized using Docker and deployed on an AWS EC2 instance.

---

## 🚀 Key Features

- **Sentence-level sentiment analysis** (Positive / Negative)
- **Model-derived confidence scoring** 
- **Word-level explainability**
  - Exact words recognized from the trained vocabulary
  - Learned weight contribution of each word
- **Numerical decision breakdown**
  - Sum of word weights
  - Intercept (model bias)
  - Sigmoid probability calculation
- **Persistent review storage** using SQLite
- **Per-review deletion** from the UI
- **Global model insights**
  - Strongest positive and negative indicators learned during training
- **Explicit handling of English-only input**
  
---

## 🧠 How Sentiment Is Determined

1. Input text is vectorized using the vocabulary learned from the IMDb dataset  
2. Only words present in the training vocabulary are retained  
3. Each retained word contributes a learned numerical weight  
4. All word weights are summed together with the model intercept  
5. The resulting score determines the sentiment direction  

A positive final score results in **Positive** sentiment,  
while a negative final score results in **Negative** sentiment.

---

## 📊 How Confidence Is Calculated

Confidence is derived **directly from the model’s probability output**.

The model applies a **sigmoid function** to the final sentence score:

Probability = 1 / (1 + e^(-score))


- If the sentence is predicted **Positive**, confidence = Positive probability  
- If the sentence is predicted **Negative**, confidence = 1 − Positive probability  

Higher absolute scores produce probabilities closer to 0% or 100%, indicating stronger confidence.
Sentences with mixed sentiment signals typically result in lower confidence values.

---

## 🛠 Tech Stack

**Application**
- Python
- Streamlit
- scikit-learn (Logistic Regression NLP model)

**Data & Persistence**
- SQLite (local persistent storage)

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

---

## 📄 License

MIT License
