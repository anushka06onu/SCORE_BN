import os
import sys

# Critical memory optimization for 512MB RAM limit
os.environ["MALLOC_ARENA_MAX"] = "2"

import time
import joblib
import warnings
import torch
import torch.nn.functional as F
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

app = FastAPI(title="SCORE-BN Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device("cpu")
CLASS_NAMES = ["General Query", "Routine", "Urgent", "Emergency"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_DIR = os.path.join(MODELS_DIR, "banglabert_onnx_quantized")

# Deep Learning Models
transformer_model = None
tokenizer = None
transformer_error = None

# Classical Models
tfidf_vectorizer = None
classical_models = {}

@app.on_event("startup")
def load_models():
    global transformer_model, tokenizer, tfidf_vectorizer, classical_models
    
    # 1. Load Transformer (ONNX INT8)
    try:
        print(f"Loading transformer model from {MODEL_DIR} onto CPU...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        transformer_model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR, file_name="model_quantized.onnx")
        print("Transformer loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading transformer: {e}", flush=True)

    # 2. Try to load Classical Models
    try:
        print("Loading classical models and TF-IDF...", flush=True)
        tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_features.joblib"))
        classical_dict = joblib.load(os.path.join(MODELS_DIR, "classical_models.joblib"))
        
        classical_models['Linear SVM'] = classical_dict.get('LinearSVM')
        classical_models['Logistic Regression'] = classical_dict.get('LogisticRegression')
        classical_models['Multinomial NB'] = classical_dict.get('MultinomialNB')
        classical_models['Random Forest'] = classical_dict.get('RandomForest')
        classical_models['XGBoost'] = classical_dict.get('XGBoost')
        
        classical_models['Tuned Linear SVM'] = joblib.load(os.path.join(MODELS_DIR, "tuned_linear_svm.joblib"))
        classical_models['Tuned XGBoost'] = joblib.load(os.path.join(MODELS_DIR, "tuned_xgboost.joblib"))
        
        print("Classical models loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading classical models: {e}", flush=True)

class PredictRequest(BaseModel):
    text: str
    model: str = "BanglaBERT"

class ExplainRequest(BaseModel):
    text: str
    model: str = "BanglaBERT"

def predict_probabilities(texts, model_name="BanglaBERT"):
    if model_name in ["BanglaBERT", "SCORE-BN", "CNN", "BiGRU", "BiLSTM"]:
        if transformer_model is None or tokenizer is None:
            raise ValueError("Transformer model not loaded")
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = transformer_model(**inputs)
            probs = F.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
        return probs
    else:
        if not tfidf_vectorizer or model_name not in classical_models or classical_models[model_name] is None:
            raise ValueError(f"Model {model_name} not loaded")
        features = tfidf_vectorizer.transform(texts)
        model_instance = classical_models[model_name]
        if hasattr(model_instance, "predict_proba"):
            try:
                probs = model_instance.predict_proba(features)
            except AttributeError:
                # Fallback for models without predict_proba that throw an AttributeError (e.g. old LogisticRegression)
                decision = model_instance.decision_function(features)
                import numpy as np
                exp_dec = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                probs = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
        else:
            # Fallback for models strictly without predict_proba (like SVC)
            decision = model_instance.decision_function(features)
            import numpy as np
            exp_dec = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probs = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
        return probs

@app.get("/api/health")
def health_check():
    return {"status": "ok", "transformer_loaded": transformer_model is not None, "classical_loaded": tfidf_vectorizer is not None}

@app.post("/api/predict")
def predict(request: PredictRequest):
    start_time = time.time()
    try:
        probs = predict_probabilities([request.text], request.model)[0]
        predicted_idx = int(probs.argmax())
        
        probabilities_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "predicted_class_id": predicted_idx,
            "predicted_class": CLASS_NAMES[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "probabilities": probabilities_dict,
            "model_used": request.model,
            "low_confidence": float(probs[predicted_idx]) < 0.5,
            "processing_time_ms": processing_time_ms,
            "disclaimer": "Research demonstration only; not a medical diagnosis."
        }
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/explain")
def explain(request: ExplainRequest):
    try:
        explainer = LimeTextExplainer(class_names=CLASS_NAMES, split_expression=r'\s+')
        
        def custom_predict(texts):
            return predict_probabilities(texts, request.model)
        
        probs = custom_predict([request.text])[0]
        top_class_idx = int(probs.argmax())
        
        exp = explainer.explain_instance(
            request.text, 
            custom_predict, 
            num_features=10, 
            num_samples=100,
            labels=(top_class_idx,)
        )
        
        explanation_list = exp.as_list(label=top_class_idx)
        
        return {
            "success": True,
            "predicted_class": CLASS_NAMES[top_class_idx],
            "explanation": [{"word": word, "weight": float(weight)} for word, weight in explanation_list]
        }
    except Exception as e:
        print("Explanation error:", e)
        raise HTTPException(status_code=500, detail=str(e))
