# src/app_streamlit.py
import streamlit as st
from predict import load_model, predict
import numpy as np
import traceback

PRIMARY = "#df1e8e"
TEXT = "#000000"
WHITE = "#ffffff"

st.set_page_config(page_title="AI Sentence Detector", layout="centered")

# CSS styling
st.markdown(
    f"""
    <style>
    .stApp {{ background: #ffffff; color: {TEXT}; }}
    .title {{ color: {PRIMARY}; font-size:28px; font-weight:700; }}
    .highlight {{ padding:2px 6px; border-radius:6px; margin:2px; display:inline-block; }}

    div.stButton > button {{
        background: linear-gradient(90deg, #6c3bf2 0%, #df1e8e 100%);
        color: {WHITE};
        border: none;
        border-radius: 50px;
        padding: 10px 34px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 6px 16px rgba(223,30,142,0.18);
        transition: transform 0.12s ease, opacity 0.12s ease;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        opacity: 0.95;
    }}

    div.stButton > button[title] {{
        padding: 8px 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>✨ Sentence Transformation Detector AI</div>", unsafe_allow_html=True)

# ✅ Cached model loading (from local or Hugging Face)
@st.cache_resource
def load():
    try:
        return load_model()
    except Exception as e:
        st.error("❌ Failed to load model. Check logs.")
        print(traceback.format_exc())
        return None, None, None, None

model, tokenizer, label_encoder, device = load()

# session state
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.result = None
    st.session_state.last_sentence = ""

sentence = st.text_area(
    "Enter a transformed sentence :",
    value=st.session_state.last_sentence or "",
    height=140
)

col1, col2 = st.columns([1, 1])
analyze_clicked = col1.button("Analyze", key="analyze")
explain_clicked = col2.button("Explain Prediction", key="explain")

# ANALYZE action
if analyze_clicked:
    if sentence.strip():
        with st.spinner("Analyzing…"):
            try:
                result = predict(
                    sentence, model, tokenizer, label_encoder, device, return_attention=True
                )
                st.session_state.analyzed = True
                st.session_state.result = result
                st.session_state.last_sentence = sentence
            except Exception as e:
                st.session_state.analyzed = False
                st.error("Error during prediction. See logs for details.")
                print(traceback.format_exc())
    else:
        st.warning("⚠️ Please enter a sentence.")

# Results display
if st.session_state.analyzed and st.session_state.result is not None:
    res = st.session_state.result
    st.success(f"Predicted: **{res['label']}** ({res['confidence']*100:.2f}% confidence)")

    for cls, p in zip(label_encoder.classes_, res["probs"]):
        st.write(f"- {cls}: {p*100:.2f}%")

    st.markdown("**Token importance (from attention):**")
    attn = res.get("attention")
    if attn:
        toks, scores = zip(*attn)
        scores = np.array(scores)
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        spans = []
        for tok, s in zip(toks, norm):
            display_tok = tok.replace("##", "")
            bg = f"rgba(223, 30, 142, {0.15 + s*0.6})"
            spans.append(f"<span class='highlight' style='background:{bg};'>{display_tok}</span>")
        st.markdown(" ".join(spans), unsafe_allow_html=True)

    # SHAP explanation
    if explain_clicked:
        with st.spinner("Generating SHAP explanation…"):
            try:
                import shap
                import torch

                def f(texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    enc = tokenizer(
                        list(texts),
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=128,
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.no_grad():
                        out = model(**enc)
                        return out.logits.cpu().numpy()

                masker = shap.maskers.Text(" ")
                explainer = shap.Explainer(f, masker)
                shap_values = explainer([st.session_state.last_sentence])

                html = shap.plots.text(shap_values[0], display=False)
                st.components.v1.html(html, height=420, scrolling=True)
            except Exception as e:
                st.error("❌ SHAP explanation failed. See logs.")
                print(traceback.format_exc())
