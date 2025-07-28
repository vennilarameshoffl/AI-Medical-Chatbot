import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("C:/Users/venni/Downloads/archive/dataset.csv")
desc_df = pd.read_csv("C:/Users/venni/Downloads/archive/symptom_Description.csv")
prec_df = pd.read_csv("C:/Users/venni/Downloads/archive/symptom_precaution.csv")

symptom_cols = [col for col in df.columns if col.startswith('Symptom')]
df[symptom_cols] = df[symptom_cols].fillna('')
df["Symptoms"] = df[symptom_cols].values.tolist()
df["Symptoms"] = df["Symptoms"].apply(lambda x: [sym.strip() for sym in x if sym.strip() != ''])

le = LabelEncoder()
df["Disease_Label"] = le.fit_transform(df["Disease"])

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])
y = df["Disease_Label"]

@st.cache_resource
def train_model():
    model = MultinomialNB()
    model.fit(X, y)
    return model

model = train_model()
all_symptoms = sorted(mlb.classes_)

def predict_disease(symptoms):
    input_vector = np.zeros(len(all_symptoms))
    for s in symptoms:
        if s in all_symptoms:
            input_vector[all_symptoms.index(s)] = 1
    pred = model.predict([input_vector])[0]
    prob = model.predict_proba([input_vector]).max()
    disease = le.inverse_transform([pred])[0]
    return disease, prob

def get_description(disease_name):
    match = desc_df[desc_df['Disease'].str.lower().str.strip() == disease_name.lower().strip()]
    return match['Description'].values[0] if not match.empty else "No description found."

def get_precautions(disease_name):
    match = prec_df[prec_df['Disease'].str.lower().str.strip() == disease_name.lower().strip()]
    if not match.empty:
        return [match[f'Precaution_{i}'].values[0] for i in range(1, 5)]
    return ["No precautions found."]

st.set_page_config(page_title="AI Medical Chatbot", layout="centered")
st.title("🤖 AI Medical Chatbot - Symptom Checker")
st.markdown("Select your symptoms below and get a medical prediction:")

selected_symptoms = st.multiselect(
    "🩺 Type and select your symptoms:",
    options=all_symptoms,
    help="Start typing to filter symptoms..."
)

if st.button("🔍 Diagnose"):
    if selected_symptoms:
        disease, confidence = predict_disease(selected_symptoms)
        st.success(f"🦠 **Predicted Disease:** {disease}")
        st.info(f"📊 **Confidence Score:** {confidence:.2f}")

        st.markdown(f"**📖 Description:** {get_description(disease)}")

        precautions = get_precautions(disease)
        if precautions[0] != "No precautions found.":
            st.markdown("**🛡️ Precautions to take:**")
            for i, p in enumerate(precautions, 1):
                st.write(f"{i}. {p}")
        else:
            st.info("No specific precautions available.")
    else:
        st.warning("⚠️ Please select at least one symptom to diagnose.")

st.markdown("---")
st.markdown("### 🙋 Was this prediction helpful?")
col1, col2 = st.columns(2)
with col1:
    if st.button("👍 Yes"):
        st.success("✅ Thanks for your feedback!")
with col2:
    if st.button("👎 No"):
        st.warning("❗We'll work on improving it.")   
