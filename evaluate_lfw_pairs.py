import os
import argparse
import numpy as np
import cv2
import face_recognition
from tqdm import tqdm
from models.face_recognition_model import FaceRecModel
from models.insightface_model import InsightFaceModel
import csv

# ---- Model Loader ----
def get_model(model_name):
    if model_name == 'facerecognition':
        return lambda img: face_recognition.face_encodings(img)[0] if face_recognition.face_encodings(img) else None
    elif model_name == 'insightface':
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        def insightface_embed(img):
            faces = app.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return faces[0].embedding if faces else None
        return insightface_embed
    elif model_name == 'hybrid':
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        def hybrid_embed(img):
            faces = app.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not faces:
                return None
            box = faces[0].bbox.astype(int)
            x1, y1, x2, y2 = box
            top, right, bottom, left = y1, x2, y2, x1
            encodings = face_recognition.face_encodings(img, [(top, right, bottom, left)])
            return encodings[0] if encodings else None
        return hybrid_embed
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ---- Pair Loader ----
def load_pairs(pairs_txt):
    pairs = []
    if pairs_txt.endswith('.csv'):
        with open(pairs_txt, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                # Remove empty trailing columns
                row = [x for x in row if x.strip() != '']
                if not row or row[0].startswith('#'):
                    continue
                if len(row) == 3:
                    name, idx1, idx2 = row
                    pairs.append((True, name, int(idx1), name, int(idx2)))
                elif len(row) == 4:
                    name1, idx1, name2, idx2 = row
                    pairs.append((False, name1, int(idx1), name2, int(idx2)))
    else:
        with open(pairs_txt, 'r') as f:
            for line in f:
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                parts = line.strip().split()
                if len(parts) == 3:
                    name, idx1, idx2 = parts
                    pairs.append((True, name, int(idx1), name, int(idx2)))
                elif len(parts) == 4:
                    name1, idx1, name2, idx2 = parts
                    pairs.append((False, name1, int(idx1), name2, int(idx2)))
    return pairs

# ---- Image Loader ----
def get_image_path(lfw_dir, name, idx):
    return os.path.join(lfw_dir, name, f"{name}_{idx:04d}.jpg")

def load_image(lfw_dir, name, idx):
    path = get_image_path(lfw_dir, name, idx)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---- Evaluation ----
def compute_distance(emb1, emb2, metric='cosine'):
    if emb1 is None or emb2 is None:
        return None
    if metric == 'cosine':
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return 1 - np.dot(emb1, emb2)
    elif metric == 'euclidean':
        return np.linalg.norm(emb1 - emb2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def evaluate(pairs, lfw_dir, model_name, threshold, metric):
    model = get_model(model_name)
    y_true, y_pred, distances = [], [], []
    for is_same, name1, idx1, name2, idx2 in tqdm(pairs, desc="Evaluating pairs"):
        img1 = load_image(lfw_dir, name1, idx1)
        img2 = load_image(lfw_dir, name2, idx2)
        if img1 is None:
            print(f"[SKIP] Missing image: {get_image_path(lfw_dir, name1, idx1)}")
            continue
        if img2 is None:
            print(f"[SKIP] Missing image: {get_image_path(lfw_dir, name2, idx2)}")
            continue
        emb1 = model(img1)
        emb2 = model(img2)
        if emb1 is None:
            print(f"[SKIP] No encoding for: {get_image_path(lfw_dir, name1, idx1)}")
            continue
        if emb2 is None:
            print(f"[SKIP] No encoding for: {get_image_path(lfw_dir, name2, idx2)}")
            continue
        dist = compute_distance(emb1, emb2, metric)
        if dist is None:
            print(f"[SKIP] Could not compute distance for: {name1}, {idx1} <-> {name2}, {idx2}")
            continue
        distances.append(dist)
        y_true.append(is_same)
        if metric == 'cosine':
            y_pred.append(dist < threshold)
        else:
            y_pred.append(dist < threshold)
    return y_true, y_pred, distances

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return cm, acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate face verification on LFW pairs.")
    parser.add_argument('--lfw_dir', type=str, required=False, default='dataset', help='Path to lfw-deepfunneled directory (default: dataset/LFW)')
    parser.add_argument('--pairs_txt', type=str, required=False, default='dataset/LFW/pairs.csv', help='Path to pairs.csv (default: dataset/LFW/pairs.csv)')
    parser.add_argument('--model', type=str, choices=['facerecognition', 'insightface', 'hybrid'], default='facerecognition')
    parser.add_argument('--threshold', type=float, default=0.5, help='Distance threshold for match')
    parser.add_argument('--metric', type=str, choices=['cosine', 'euclidean'], default='cosine')
    parser.add_argument('--limit_people', type=int, default=None, help='Limit to N random people for quick evaluation (default: None)')
    args = parser.parse_args()

    # If limit_people is set, subsample people in the dataset
    if args.limit_people is not None:
        import random
        all_people = [p for p in os.listdir(args.lfw_dir) if os.path.isdir(os.path.join(args.lfw_dir, p))]
        selected_people = set(random.sample(all_people, min(args.limit_people, len(all_people))))
        # Filter pairs to only include selected people
        pairs = [pair for pair in load_pairs(args.pairs_txt) if pair[1] in selected_people and pair[3] in selected_people]
    else:
        pairs = load_pairs(args.pairs_txt)

    y_true, y_pred, distances = evaluate(pairs, args.lfw_dir, args.model, args.threshold, args.metric)
    print(f"Pairs processed: {len(y_true)} / {len(pairs)}")
    if len(y_true) == 0:
        print("No pairs were successfully processed. Check your image paths and encoding steps.")
        return
    cm, acc = compute_metrics(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc*100:.2f}%")

    # Optional: ROC curve
    try:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve([int(x) for x in y_true], distances)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
    except ImportError:
        print("matplotlib or sklearn not installed, skipping ROC curve.")

if __name__ == "__main__":
    main()