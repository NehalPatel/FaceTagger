import os
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import face_recognition
from tqdm import tqdm

# Optional: import insightface only if needed
try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

def get_people_images(lfw_dir):
    people = [p for p in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, p))]
    people_images = {}
    for person in people:
        imgs = [f for f in os.listdir(os.path.join(lfw_dir, person)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(imgs) >= 2:
            people_images[person] = imgs
    return people_images

def encode_gallery_face_recognition(lfw_dir, gallery_dict):
    encodings = {}
    for person, img in tqdm(gallery_dict.items(), desc="Encoding gallery (face_recognition)"):
        img_path = os.path.join(lfw_dir, person, img)
        image = face_recognition.load_image_file(img_path)
        faces = face_recognition.face_encodings(image)
        if faces:
            encodings[person] = faces[0]
    return encodings

def encode_gallery_insightface(lfw_dir, gallery_dict, app):
    encodings = {}
    for person, img in tqdm(gallery_dict.items(), desc="Encoding gallery (insightface)"):
        img_path = os.path.join(lfw_dir, person, img)
        image = cv2.imread(img_path)
        faces = app.get(image)
        if faces:
            encodings[person] = faces[0].embedding
    return encodings

def recognize_face_recognition(gallery_encodings, gallery_names, probe_encoding, threshold=0.6):
    if probe_encoding is None:
        return 'Unknown', False
    matches = face_recognition.compare_faces(gallery_encodings, probe_encoding, tolerance=threshold)
    if True in matches:
        idx = matches.index(True)
        return gallery_names[idx], True
    return 'Unknown', False

def recognize_insightface(gallery_encodings, gallery_names, probe_encoding, threshold=1.2, metric='l2'):
    if probe_encoding is None:
        return 'Unknown', False
    # Normalize embeddings
    probe_encoding = probe_encoding / np.linalg.norm(probe_encoding)
    gallery_encodings = [enc / np.linalg.norm(enc) for enc in gallery_encodings]
    if metric == 'l2':
        dists = [np.linalg.norm(probe_encoding - enc) for enc in gallery_encodings]
        if not dists:
            return 'Unknown', False
        min_idx = np.argmin(dists)
        if dists[min_idx] < threshold:
            return gallery_names[min_idx], True
        return 'Unknown', False
    elif metric == 'cosine':
        sims = [np.dot(probe_encoding, enc) for enc in gallery_encodings]
        max_idx = np.argmax(sims)
        if sims[max_idx] > (1 - threshold):
            return gallery_names[max_idx], True
        return 'Unknown', False
    else:
        print(f"[WARN] Unknown metric: {metric}. Using l2 by default.")
        dists = [np.linalg.norm(probe_encoding - enc) for enc in gallery_encodings]
        if not dists:
            return 'Unknown', False
        min_idx = np.argmin(dists)
        if dists[min_idx] < threshold:
            return gallery_names[min_idx], True
        return 'Unknown', False

def recognize_hybrid(app, gallery_dict, gallery_encodings, gallery_names, lfw_dir, person, img, threshold=0.6):
    img_path = os.path.join(lfw_dir, person, img)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(img_bgr)
    if not faces:
        return False, 'Unknown', False
    box = faces[0].bbox.astype(int)
    x1, y1, x2, y2 = box
    top, right, bottom, left = y1, x2, y2, x1
    encodings = face_recognition.face_encodings(img_rgb, [(top, right, bottom, left)])
    if not encodings:
        return True, 'Unknown', False
    probe_encoding = encodings[0]
    pred, found = recognize_face_recognition(gallery_encodings, gallery_names, probe_encoding, threshold)
    return True, pred, found

def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition models on LFW gallery/probe split.")
    parser.add_argument('--lfw_dir', type=str, default='dataset/LFW/lfw-deepfunneled/lfw-deepfunneled', help='Path to LFW dataset (default: dataset/LFW/lfw-deepfunneled/lfw-deepfunneled)')
    parser.add_argument('--num_people', type=int, default=None, help='Number of people to sample (default: all qualified)')
    parser.add_argument('--models', type=str, nargs='+', default=['facerecognition', 'insightface', 'hybrid'], choices=['facerecognition', 'insightface', 'hybrid'])
    parser.add_argument('--output_csv', type=str, default=None, help='Optional: path to save results as CSV')
    parser.add_argument('--metric', type=str, default='l2', choices=['l2', 'cosine'], help='Metric for insightface: l2 or cosine (default: l2)')
    parser.add_argument('--threshold', type=float, default=None, help='Threshold for insightface (default: 1.2 for l2, 0.35 for cosine)')
    args = parser.parse_args()

    # Step 1: Prepare gallery/probe split
    people_images = get_people_images(args.lfw_dir)
    qualified_people = [p for p in people_images.keys() if len(people_images[p]) >= 2]
    n_qualified = len(qualified_people)
    if n_qualified == 0:
        print("No people with >=2 images found in the dataset.")
        return
    if args.num_people is None or args.num_people > n_qualified:
        selected_people = qualified_people
        print(f"Using all {n_qualified} qualified people (with >=2 images).")
    else:
        selected_people = random.sample(qualified_people, args.num_people)
        print(f"Using {args.num_people} randomly selected qualified people (out of {n_qualified}).")
    gallery_dict = {}
    probe_dict = {}
    for person in selected_people:
        imgs = people_images[person]
        random.shuffle(imgs)
        gallery_dict[person] = imgs[0]
        probe_dict[person] = imgs[1:]

    # Step 2: Encode gallery for each model
    results = []
    if 'facerecognition' in args.models or 'hybrid' in args.models:
        gallery_encodings_fr = []
        gallery_names_fr = []
        enc_fr = encode_gallery_face_recognition(args.lfw_dir, gallery_dict)
        for person in gallery_dict:
            if person in enc_fr:
                gallery_encodings_fr.append(enc_fr[person])
                gallery_names_fr.append(person)
    if 'insightface' in args.models or 'hybrid' in args.models:
        if FaceAnalysis is None:
            raise ImportError("insightface not installed!")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        gallery_encodings_if = []
        gallery_names_if = []
        enc_if = encode_gallery_insightface(args.lfw_dir, gallery_dict, app)
        for person in gallery_dict:
            if person in enc_if:
                gallery_encodings_if.append(enc_if[person])
                gallery_names_if.append(person)

    # Step 3: Evaluate probe images
    total_probes = sum(len(probe_dict[p]) for p in selected_people)
    # Set default threshold if not provided
    if args.threshold is None:
        if args.metric == 'l2':
            insightface_threshold = 1.2
        else:
            insightface_threshold = 0.35
    else:
        insightface_threshold = args.threshold
    with tqdm(total=total_probes, desc="Processing probe images") as pbar:
        for person in selected_people:
            for img in probe_dict[person]:
                img_path = os.path.join(args.lfw_dir, person, img)
                row = {'person': person, 'img': img}
                # face_recognition
                if 'facerecognition' in args.models:
                    image = face_recognition.load_image_file(img_path)
                    faces = face_recognition.face_encodings(image)
                    row['fr_detected'] = bool(faces)
                    if faces:
                        pred, found = recognize_face_recognition(gallery_encodings_fr, gallery_names_fr, faces[0])
                        row['fr_pred'] = pred
                        row['fr_correct'] = (pred == person)
                    else:
                        row['fr_pred'] = 'Unknown'
                        row['fr_correct'] = False
                # insightface
                if 'insightface' in args.models:
                    faces = app.get(cv2.imread(img_path))
                    row['if_detected'] = bool(faces)
                    if faces:
                        pred, found = recognize_insightface(gallery_encodings_if, gallery_names_if, faces[0].embedding, threshold=insightface_threshold, metric=args.metric)
                        row['if_pred'] = pred
                        row['if_correct'] = (pred == person)
                    else:
                        row['if_pred'] = 'Unknown'
                        row['if_correct'] = False
                # hybrid
                if 'hybrid' in args.models:
                    detected, pred, found = recognize_hybrid(app, gallery_dict, gallery_encodings_fr, gallery_names_fr, args.lfw_dir, person, img)
                    row['hy_detected'] = detected
                    row['hy_pred'] = pred
                    row['hy_correct'] = (pred == person)
                results.append(row)
                pbar.update(1)

    df = pd.DataFrame(results)
    # Step 4: Compute metrics
    summary = []
    for model in args.models:
        if model == 'facerecognition':
            detected = df['fr_detected'].sum()
            total = len(df)
            correct = df['fr_correct'].sum()
            precision = correct / detected if detected else 0
            recall = correct / total if total else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            summary.append({'Model': 'face_recognition', 'Detection Rate': detected/total, 'Accuracy': correct/total, 'Precision': precision, 'Recall': recall, 'F1': f1})
        if model == 'insightface':
            detected = df['if_detected'].sum()
            total = len(df)
            correct = df['if_correct'].sum()
            precision = correct / detected if detected else 0
            recall = correct / total if total else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            summary.append({'Model': 'insightface', 'Detection Rate': detected/total, 'Accuracy': correct/total, 'Precision': precision, 'Recall': recall, 'F1': f1})
        if model == 'hybrid':
            detected = df['hy_detected'].sum()
            total = len(df)
            correct = df['hy_correct'].sum()
            precision = correct / detected if detected else 0
            recall = correct / total if total else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            summary.append({'Model': 'hybrid', 'Detection Rate': detected/total, 'Accuracy': correct/total, 'Precision': precision, 'Recall': recall, 'F1': f1})
    summary_df = pd.DataFrame(summary)
    print("\n=== Model Comparison Matrix ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Detailed results saved to {args.output_csv}")

if __name__ == "__main__":
    main()