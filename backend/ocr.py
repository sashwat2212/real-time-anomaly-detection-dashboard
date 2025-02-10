import pytesseract
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
import pdf2image
import os
from typing import Union

def preprocess_image(image: np.ndarray) -> np.ndarray:
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # grayscale

    
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2) # adaptive thresholding

    # deskew the image
    coords = np.column_stack(np.where(processed > 0)) 
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = processed.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    processed = cv2.warpAffine(processed, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Deblur the image
    processed = cv2.GaussianBlur(processed, (3, 3), 0)

    return processed

def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Unable to read image file."
        
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image)
        return text.strip()
    except Exception as e:
        return f"Error during OCR processing: {str(e)}"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages of a PDF."""
    try:
        images = pdf2image.convert_from_path(pdf_path)
        extracted_text = []
        
        for i, img in enumerate(images):
            # convert to opencv format
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            processed_image = preprocess_image(img_cv)
            text = pytesseract.image_to_string(processed_image)
            extracted_text.append(text.strip())

        return "\n\n".join(extracted_text)
    except Exception as e:
        return f"Error during PDF OCR processing: {str(e)}"

def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        return extract_text_from_image(file_path)
    elif file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        return "Unsupported file format. Please provide an image or PDF."



