# app.py

# ===== Imports =====
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ===== Data + feature engineering =====
@st.cache_data
def load_data():
    s = pd.read_csv("social_media_usage.csv")  # use relative path for deployment

    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    ss = pd.DataFrame({
        "sm_li": clean_sm(s["web1h"]),  # LinkedIn user target
        "income": np.where(s["income"].between(1, 9), s["income"], np.nan),
        "education": np.where(s["educ2"].between(1, 8), s["educ2"], np.nan),
        "parent": clean_sm(s["par"]),
        "married": clean_sm(s["marital"]),
        "female": np.where(s["gender"] == 2, 1, 0),
        "age": np.where(s["age"] <= 98, s["age"], np.nan),
    })

    ss = ss.dropna()
    return ss

ss = load_data()

X = ss[["income", "education", "parent", "married", "female", "age"]]
y = ss["sm_li"]

# ===== Train model =====
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=987,
    )

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=2000
    )
    lr.fit(X_train, y_train)
    return lr

lr = train_model(X, y)

# ===== Streamlit UI =====
st.title("LinkedIn Usage Prediction App")

st.write("Move the sliders / selectors to see how the predicted probability changes.")

income = st.slider("Income (1–9)", 1, 9, 8)
education = st.slider("Education (1–8)", 1, 8, 7)
parent = st.selectbox("Parent of child under 18?", ["No", "Yes"])
married = st.selectbox("Married?", ["No", "Yes"])
female = st.selectbox("Female?", ["No", "Yes"])
age = st.slider("Age", 18, 98, 42)

parent_bin = 1 if parent == "Yes" else 0
married_bin = 1 if married == "Yes" else 0
female_bin = 1 if female == "Yes" else 0

newdata = pd.DataFrame([{
    "income": income,
    "education": education,
    "parent": parent_bin,
    "married": married_bin,
    "female": female_bin,
    "age": age,
}])

# Predict
pred_class = lr.predict(newdata)[0]
prob = lr.predict_proba(newdata)[0, 1]

st.write(f"**Predicted class:** {pred_class}  (0 = not LinkedIn user, 1 = LinkedIn user)")
st.write(f"**Estimated probability of using LinkedIn:** {prob:.2%}")

# Bar chart of probabilities
prob_df = pd.DataFrame({
    "class": ["LinkedIn user", "Not LinkedIn user"],
    "probability": [prob, 1 - prob],
})

chart = (
    alt.Chart(prob_df)
    .mark_bar()
    .encode(
        x=alt.X("class:N", title="Class"),
        y=alt.Y("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
        tooltip=["class", alt.Tooltip("probability:Q", format=".2%")],
    )
    .interactive()
)

st.altair_chart(chart, use_container_width=True)
