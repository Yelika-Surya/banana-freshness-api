# Banana Freshness Predictor

Predicts remaining shelf life of a banana from a photo using EfficientNet-B0 + FastAPI.

## Classes
| Class | Label        | Days Remaining |
|-------|-------------|----------------|
| a     | Green       | 10 – 14 days   |
| b     | Half-ripe   | 6 – 9 days     |
| c     | Ripe        | 2 – 5 days     |
| d     | Rotten      | 0 – 1 days     |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Structure
```
dataset/
├── train/
│   ├── a/
│   ├── b/
│   ├── c/
│   └── d/
├── val/
│   ├── a/ b/ c/ d/
└── test/
    ├── a/ b/ c/ d/
```

## 1. Train

```bash
python train.py \
  --data_root ./dataset \
  --save_dir  ./models \
  --epochs    25 \
  --batch_size 32 \
  --lr        0.0001
```

Outputs in `./models/`:
- `banana_model.pth`     ← trained weights + metadata
- `metadata.json`        ← classes, days map, accuracy
- `training_curves.png`  ← loss & accuracy plots
- `confusion_matrix.png` ← test set confusion matrix

GPU recommended. On CPU, reduce `--batch_size 16`.

## 2. Test a single image

```bash
python predict.py ./models/banana_model.pth ./test_banana.jpg
```

## 3. Run the API

```bash
# Set model path (optional, defaults to ./models/banana_model.pth)
export MODEL_PATH=./models/banana_model.pth

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

## 4. API Usage

### Predict from file
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@banana.jpg"
```

### Response
```json
{
  "filename": "banana.jpg",
  "class": "b",
  "label": "Half-ripe",
  "confidence": 0.87,
  "days": {
    "estimated": 7.2,
    "min": 6,
    "max": 9,
    "message": "Half-ripe. Will be at peak in a few days (~7 days remaining)."
  },
  "probabilities": {
    "a": 0.02,
    "b": 0.87,
    "c": 0.10,
    "d": 0.01
  }
}
```

### Health check
```bash
curl http://localhost:8000/health
```

### List classes
```bash
curl http://localhost:8000/classes
```

## How days prediction works

Rather than just returning the predicted class, the API computes a weighted average
of all class midpoints using softmax probabilities:

```
days = sum(prob[cls] * midpoint[cls] for cls in classes)
```

This gives a smooth, continuous estimate. For example:
- 60% half-ripe (mid=7) + 40% ripe (mid=3) → **5.4 days**

## Tips for better accuracy
- Ensure even class distribution across train/val/test
- Photograph bananas in good lighting, against a plain background
- Include photos from multiple angles per banana
