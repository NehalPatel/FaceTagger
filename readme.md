# 🧠 FaceTagger

FaceTagger is a simple Python-based face recognition system that can identify known faces in images using deep learning. Built on top of the `face_recognition` library and OpenCV, this project demonstrates how to build a modular pipeline for face recognition.

---

## 📸 Features

- Detect and recognize faces from images
- Label known faces using image file names
- Easily add new known persons
- Works offline, lightweight, and extensible
- Can be extended for webcam/video or real-time surveillance

---

## 📁 Project Structure
FaceTagger/
├── known_faces/ # Images of known people (one face per image)
├── test_images/ # Images with multiple unknown faces
├── encodings/ # Pickled face encodings
├── app/
│ ├── encoder.py # Encodes new faces
│ ├── recognizer.py # Recognizes faces in test images
│ └── utils.py # Utility/helper functions
├── main.py # Entry point to test recognition
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/FaceTagger.git
cd FaceTagger
```

### 2. Set up a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# OR
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 🧪 How It Works
1. Place images of known people in the known_faces/ folder (e.g., nehal.jpg)

2. Place a group photo in the test_images/ folder

3. Run the app:
```bash
python main.py
```

4. The system detects faces in the test image, compares them with known encodings, and labels them.

#### 🛠️ Dependencies
face_recognition

opencv-python

Python 3.7+

#### 🧩 Future Enhancements
* Add real-time webcam recognition
* Use a database for face storage and logging
* Integrate with Flask/Django for web-based interface
* Train on your own face dataset
* Add confidence threshold & face clustering

#### 👨‍💻 Author
Nehal Patel
iamnehalpatel@gmail.com
Professor | Developer | AI Researcher
📍 Surat, Gujarat, India

#### 🛡️ License
This project is open-source and available under the MIT License.