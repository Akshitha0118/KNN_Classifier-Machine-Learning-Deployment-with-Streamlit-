import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from matplotlib.colors import ListedColormap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="KNN Classification Dashboard",
    layout="wide"
)

# ---------------- LOAD CSS ----------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- LOAD MODEL ----------------
with open("KNN_Classifier.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- TITLE ----------------
st.markdown("<div class='main-title'>📊 K-Nearest Neighbors Classification</div>", unsafe_allow_html=True)

# ---------------- DATA UPLOAD ----------------
st.markdown("<div class='sub-title'>1️⃣ Upload Dataset</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    X = data.iloc[:, [2, 3]].values
    y = data.iloc[:, -1].values

    # ---------------- SCALING ----------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- PREDICTION ----------------
    st.markdown("<div class='sub-title'>2️⃣ Model Prediction</div>", unsafe_allow_html=True)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    data["Prediction"] = y_pred
    st.dataframe(data.head())

    # ---------------- METRICS ----------------
    st.markdown("<div class='sub-title'>3️⃣ Model Evaluation</div>", unsafe_allow_html=True)

    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='metric-box'>Accuracy<br>{:.2f}</div>".format(acc), unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-box'>AUC Score<br>{:.2f}</div>".format(auc), unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-box'>Samples<br>{}</div>".format(len(y)), unsafe_allow_html=True)

    # ---------------- CONFUSION MATRIX ----------------
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

    # ---------------- ROC CURVE ----------------
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    st.pyplot(fig2)

    # ---------------- DECISION BOUNDARY ----------------
    st.markdown("<div class='sub-title'>4️⃣ Decision Boundary</div>", unsafe_allow_html=True)

    X1, X2 = np.meshgrid(
        np.arange(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 0.01),
        np.arange(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 0.01)
    )

    fig3, ax3 = plt.subplots()
    ax3.contourf(
        X1, X2,
        model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(("red", "green"))
    )

    for i, j in enumerate(np.unique(y)):
        ax3.scatter(
            X_scaled[y == j, 0],
            X_scaled[y == j, 1],
            label=j
        )

    ax3.set_xlabel("Age")
    ax3.set_ylabel("Estimated Salary")
    ax3.legend()
    st.pyplot(fig3)
