from predict import load_model

model, tokenizer, le, device = load_model()
print("✅ Model loaded successfully")
print("Labels:", list(le.classes_))
