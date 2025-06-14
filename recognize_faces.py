import face_recognition
import cv2
import os

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir("known_faces"):
    image_path = os.path.join("known_faces", filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(filename)[0])  # 'nehal.jpg' → 'nehal'

# Load the test image
test_image = face_recognition.load_image_file("test_images/group_photo.jpeg")
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to BGR image for OpenCV display
image_to_show = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

# Recognize each face
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        match_index = matches.index(True)
        name = known_face_names[match_index]

    # Draw rectangle and label
    cv2.rectangle(image_to_show, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image_to_show, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# Show the result
cv2.imshow("Face Recognition Result", image_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
