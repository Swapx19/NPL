# Named Entity Recognition (NER) Project

## Overview
This project involves building and deploying a Named Entity Recognition (NER) model using SpaCy. The model is trained on a publicly available dataset and deployed as a REST API using FastAPI. The API takes text input and returns recognized entities.

## Project Structure
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
│── requirements.txt  # Dependencies
│── README.md  # Project Documentation
```

## Installation
Clone the repository and install the required dependencies:
```sh
git clone <repo_link>
cd ner_project
pip install -r requirements.txt
```

## Data Preprocessing
- Load the dataset (e.g., CoNLL-2003)
- Perform lowercasing, tokenization, and entity tagging
- Convert data into SpaCy-compatible format

```python
import spacy
from spacy.tokens import DocBin
from datasets import load_dataset

# Load dataset
dataset = load_dataset("conll2003")
train_data = dataset["train"]

def preprocess(text):
    text = text.lower()
    return text

train_data = [(preprocess(text), labels) for text, labels in zip(train_data["tokens"], train_data["ner_tags"])]
```

## Model Training
- Use SpaCy's NER pipeline
- Annotate entities and train the model
- Save the trained model

```python
import spacy
from spacy.training import Example

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

## Model Deployment (FastAPI)
- Load the trained model
- Create a FastAPI application with a `/predict` endpoint
- Accept text input and return recognized entities

```python
from fastapi import FastAPI, Depends, HTTPException
import spacy
from pydantic import BaseModel
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI()
security = HTTPBasic()
nlp = spacy.load("models/ner_model.pkl")

class TextRequest(BaseModel):
    text: str

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "password":
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/predict")
def predict(request: TextRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
    doc = nlp(request.text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {"entities": entities}
```

## Running the API
Start the FastAPI server:
```sh
uvicorn app.main:app --reload
```

## Docker Setup
To containerize the application, build and run the Docker image:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run Docker Container
```sh
docker build -t ner_api .
docker run -p 8000:8000 ner_api
```

## API Usage
You can test the API using `curl` or Postman:
```sh
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -u admin:password -d '{"text": "Barack Obama was the 44th President of the United States."}'
```
Expected Output:
```json
{
    "entities": [["Barack Obama", "PERSON"], ["United States", "GPE"]]
}
```

## Unit Testing
A test script is provided in `tests/test_api.py` to verify API functionality.
Run the tests using:
```sh
pytest tests/test_api.py
```

## Deployment Guide
To deploy the API on AWS/GCP/Render:
- Use a cloud service with Docker support.
- Expose port `8000` for API access.
- Store API credentials securely.

## Streamlit UI (Optional)
A simple web UI can be added using Streamlit.
Run it with:
```sh
streamlit run streamlit_app.py
```

