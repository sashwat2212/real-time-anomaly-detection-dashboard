from fastapi import FastAPI, File, UploadFile
import shutil
import os
from ocr import extract_text  # Import OCR function
from nlp_model import detect_anomaly  # Import the NLP model function


app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Document Anomaly Detection API is running!"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Perform OCR if the file is an image or PDF
    extracted_text = ""
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        extracted_text = extract_text(file_path)
        anomaly_result = detect_anomaly(extracted_text)
    elif file.filename.lower().endswith('.pdf'):
        extracted_text = extract_text(file_path)
        anomaly_result = detect_anomaly(extracted_text)

    return {
        "filename": file.filename,
        "message": "File uploaded successfully!",
        "extracted_text": extracted_text,
        "anomaly_result": anomaly_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
