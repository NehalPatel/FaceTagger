import os

lfw_dir = "dataset/LFW/lfw-deepfunneled"  # Change if needed
print("Folders in LFW dir:", os.listdir(lfw_dir)[:5])  # Show first 5 folders

# For the first folder, show how many images and their names
first_person = os.listdir(lfw_dir)[0]
person_dir = os.path.join(lfw_dir, first_person)
print("First person folder:", first_person)
print("Images in first person folder:", os.listdir(person_dir))