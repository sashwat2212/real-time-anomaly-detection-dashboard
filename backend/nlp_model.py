from transformers import pipeline

anomaly_detector = pipeline("text-classification", model="distilbert-base-uncased", truncation=True)

def detect_anomaly(text):
    
    truncated_text = text[:500]  
    result = anomaly_detector(truncated_text)
    return result
