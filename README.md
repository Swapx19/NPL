Here is your updated **README.md** with all the latest improvements:  

---

# **Named Entity Recognition (NER) API Project** 🚀  

## **Overview**  
This project involves building and deploying a Named Entity Recognition (NER) model using **SpaCy**.  
- The model is trained on a publicly available dataset.  
- Deployed as a **FastAPI REST API** with authentication.  
- Includes **Docker support**, **unit tests**, and **logging** for better maintainability.  

---

## **📁 Project Structure**
```
ner_project/
│── data/
│   ├── train.txt  # Training data
│   ├── test.txt   # Testing data
│── models/
│   ├── ner_model.pkl  # Saved model
│── notebooks/
│   ├── exploratory_data_analysis.ipynb  # EDA and Preprocessing
│   ├── model_training.ipynb  # Model Training & Evaluation
│── app/
│   ├── main.py  # FastAPI Application
│   ├── Dockerfile  # Containerization
│── tests/
│   ├── test_api.py  # Unit tests for API
│── .env  # Environment variables for authentication
│── deploy.sh  # Automated Deployment Script
│── requirements.txt  # Dependencies
│── README.md  # Project Documentation
```

---

## **💻 Installation**
Clone the repository and install the required dependencies:  
```sh
git clone <repo_link>
cd ner_project
pip install -r requirements.txt
```

---

## **📊 Data Preprocessing**
- Dataset used: **CoNLL-2003**  
- Preprocessing includes **tokenization, stopword removal, and entity tagging**  
- Converts data into **SpaCy-compatible format**  

Example preprocessing code:  
```python
import spacy
from datasets import load_dataset

# Load dataset
dataset = load_dataset("conll2003")
train_data = dataset["train"]

def preprocess(text):
    return text.lower()

train_data = [(preprocess(text), labels) for text, labels in zip(train_data["tokens"], train_data["ner_tags"])]
```

---

## **🤖 Model Training**
- Uses **SpaCy’s NER pipeline**  
- Annotates entities and trains the model  
- Saves the trained model  

Example training snippet:  
```python
import spacy
from spacy.training import Example
from spacy.tokens import DocBin

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

for _, annotations in train_data:
    for ent in annotations:
        ner.add_label(ent)

doc_bin = DocBin()
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, {"entities": annotations})
    doc_bin.add(example)

doc_bin.to_disk("train.spacy")
```

---

## **🚀 Model Deployment (FastAPI)**
- Loads the **trained model**  
- Creates a FastAPI application  
- Implements **basic authentication**  
- Includes **error handling & logging**  

Example **API Code (app/main.py)**:  
```python
import logging
import os
import spacy
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBasic()

# Load Model
nlp = spacy.load("models/ner_model.pkl")

# Authentication using Environment Variables
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")

class TextRequest(BaseModel):
    text: str

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != ADMIN_USERNAME or credentials.password != ADMIN_PASSWORD:
        logger.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/predict")
def predict(request: TextRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")
    
    doc = nlp(request.text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    logger.info(f"Prediction successful: {entities}")
    return {"entities": entities}
```

---

## **⚙ Running the API**
### **Local Deployment**
Start the FastAPI server:  
```sh
uvicorn app.main:app --reload
```
Test the API:  
```sh
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -u admin:password -d '{"text": "Barack Obama was the 44th President of the United States."}'
```
**Expected Output**:  
```json
{
    "entities": [["Barack Obama", "PERSON"], ["United States", "GPE"]]
}
```

---

## **🐳 Docker Setup**
### **Dockerfile**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
### **Build and Run Docker Container**
```sh
docker build -t ner_api .
docker run -p 8000:8000 ner_api
```

---

## **🧪 Unit Testing**
A test script is provided in `tests/test_api.py`.  
Run the tests using:
```sh
pytest tests/test_api.py
```

### **Example Test Cases**
```python
import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "http://127.0.0.1:8000"

def test_valid_request():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": "Elon Musk founded SpaceX."},
        auth=HTTPBasicAuth("admin", "password"),
    )
    assert response.status_code == 200
    assert "entities" in response.json()

def test_invalid_auth():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": "Elon Musk founded SpaceX."},
        auth=HTTPBasicAuth("wrong_user", "wrong_pass"),
    )
    assert response.status_code == 401

def test_empty_text():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": ""},
        auth=HTTPBasicAuth("admin", "password"),
    )
    assert response.status_code == 422

if __name__ == "__main__":
    test_valid_request()
    test_invalid_auth()
    test_empty_text()
    print("All tests passed.")
```

---

## **📦 Automated Deployment**
A **deployment script** is included for **Docker-based deployment**.

### **`deploy.sh`**
```sh
#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

echo "Building Docker image..."
docker build -t ner_api .

echo "Running Docker container..."
docker run -p 8000:8000 --env ADMIN_USERNAME=$ADMIN_USERNAME --env ADMIN_PASSWORD=$ADMIN_PASSWORD ner_api
```
Make it executable:
```sh
chmod +x deploy.sh
```
Run:
```sh
./deploy.sh
```

---

## **☁️ Cloud Deployment Guide**
- Deploy on **AWS / GCP / Render**  
- Expose **port 8000**  
- Use **environment variables** for authentication  

---

