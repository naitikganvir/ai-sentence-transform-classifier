[📑 Sentence Transformation Classification

🔹 Problem Statement
The task is to build a AI ML model that can
detect sentence transformation types such as:

1)Active → Passive
2)Passive → Active
3)Direct → Indirect Speech
4)Indirect → Direct Speech
5)Positive → Negative
6)Negative → Positive

This model should classify the transformation applied between 
an original sentence and its transformed sentence.

🔹 Dataset Description
Manually created dataset of sentence pairs.

Each record contains:

Original Sentence

Transformed Sentence

Label (type of transformation)

📌 Example:

ID	Original Sentence	Transformed Sentence	Label
1	The company will launch the new product next quarter	The new product will be launched by the company next quarter
Active → Passive


🔹 Model Details
Model Type: Transformer-based classifier (BERT).
Task: Multi-class classification (6 transformation types).

Frameworks:
PyTorch, Pandas, Numpy, ScikitLearn, Streamlit, Matplotlib

HuggingFace Transformers (BERT) 

Shap for XAI

Scikit-learn

🔹 Training & Evaluation

Metrics used:

Accuracy Score

Evaluation performed using confusion matrix analysis.


🔹 Explainability

Used SHAP to explain predictions.
Highlights words that influenced classification.

Example:
Input: "The teacher said, 'You must study hard.'"

Prediction: Direct → Indirect Speech

SHAP shows influence of 'said' and quotation marks.

🔹 Error Analysis
Observed confusions between:
structural changes look similar, So the model get Confused sometime

🔹 Deployment 

Lightweight app built using Streamlit.

Features:

Upload sentence pairs
Predict transformation type
Explain predictions with SHAP

Run with:

streamlit run src/app_streamlit.py]()