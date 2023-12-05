import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
 
import os
import random
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
 
 
import os
import random
 
def load_and_split_dataset(folder_path, test_ratio=0.2):
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]
    random.shuffle(image_paths)
 
    num_test = int(len(image_paths) * test_ratio)
    test_image_paths = image_paths[:num_test]
    train_image_paths = image_paths[num_test:]
 
    return train_image_paths, test_image_paths
 
def extract_features(image_paths, feature_detector):
    features = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, des = feature_detector.detectAndCompute(image, None)
        if des is not None:
            features.extend(des)
    return np.array(features)
 
def create_codebook(features, num_dict_features):
    kmeans = KMeans(n_clusters=num_dict_features)
    kmeans.fit(features)
    return kmeans.cluster_centers_
 
def image_representation(image_paths, codebook, feature_detector):
    representations = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, des = feature_detector.detectAndCompute(image, None)
        if des is not None:
            histogram = np.zeros(len(codebook))
            for feature in des:
                idx = np.argmin(np.linalg.norm(codebook - feature, axis=1))
                histogram[idx] += 1
            representations.append(histogram)
    return np.array(representations)
 
def train_model(train_image_paths, test_image_paths, feature_detector, num_dict_features):
    # Extract features from training images
    train_features = extract_features(train_image_paths, feature_detector)
 
    # Create codebook using KMeans clustering
    codebook = create_codebook(train_features, num_dict_features)
 
    # Represent training and testing images using BoVW
    train_data = image_representation(train_image_paths, codebook, feature_detector)
    test_data = image_representation(test_image_paths, codebook, feature_detector)
 
    # Assuming you have labels for training images (you may need to modify this part)
    train_labels = [get_label_from_path(path) for path in train_image_paths]
    test_labels = [get_label_from_path(path) for path in test_image_paths]
 
    # Train an SVM model
    svm_model = SVC()
    svm_model.fit(train_data, train_labels)
 
    # Make predictions on the test set
    predictions = svm_model.predict(test_data)
 
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
 
    return svm_model, accuracy
 
# Example function to extract label from image path (modify as needed)
def get_label_from_path(image_path):
    # Modify this function based on your dataset structure
    return os.path.basename(os.path.dirname(image_path))
 
# Example usage
dataset_path = "DATASET/"
train_image_paths, test_image_paths = load_and_split_dataset(dataset_path, test_ratio=0.2)
 
# The rest of the code remains unchanged...
feature_detector = cv2.ORB_create()
num_dict_features = 50  # Parameter num_dict_features
 
trained_model, accuracy = train_model(train_image_paths, test_image_paths, feature_detector, num_dict_features)
 
print(f"Model Accuracy: {accuracy}")
 
def evaluate_model_on_subset(image_paths, feature_detector, num_dict_features, trained_model):
    # Represent images using BoVW
    features = extract_features(image_paths, feature_detector)
    codebook = create_codebook(features, num_dict_features)
    data = image_representation(image_paths, codebook, feature_detector)
 
    # Assuming you have labels for images (you may need to modify this part)
    labels = [get_label_from_path(path) for path in image_paths]
 
    # Make predictions
    predictions = trained_model.predict(data)
 
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
 
    return accuracy
 
 
autumn_image_paths, winter_image_paths, spring_image_paths = [], [], []
autumn_day_image_paths, autumn_night_image_paths = [], []
winter_day_image_paths, winter_night_image_paths = [], []
spring_day_image_paths, spring_night_image_paths = [], []
 
for image_path in train_image_paths:
    # Assuming your images are named with the specified convention: <label>_<season_and_time_of_day>_<numerical_index>.jpg
    label, season_and_time_of_day, _ = os.path.splitext(os.path.basename(image_path))[0].split('_')
 
    if label == "00" and season_and_time_of_day == "00":
        autumn_day_image_paths.append(image_path)
    elif label == "00" and season_and_time_of_day == "01":
        autumn_night_image_paths.append(image_path)
    elif label == "02" and season_and_time_of_day == "02":
        winter_day_image_paths.append(image_path)
    elif label == "02" and season_and_time_of_day == "03":
        winter_night_image_paths.append(image_path)
    elif label == "04" and season_and_time_of_day == "04":
        spring_day_image_paths.append(image_path)
    elif label == "04" and season_and_time_of_day == "05":
        spring_night_image_paths.append(image_path)
 
# Evaluate on subsets
autumn_accuracy = evaluate_model_on_subset(autumn_image_paths, feature_detector, num_dict_features, trained_model)
winter_accuracy = evaluate_model_on_subset(winter_image_paths, feature_detector, num_dict_features, trained_model)
spring_accuracy = evaluate_model_on_subset(spring_image_paths, feature_detector, num_dict_features, trained_model)
 
autumn_day_accuracy = evaluate_model_on_subset(autumn_day_image_paths, feature_detector, num_dict_features, trained_model)
autumn_night_accuracy = evaluate_model_on_subset(autumn_night_image_paths, feature_detector, num_dict_features, trained_model)
winter_day_accuracy = evaluate_model_on_subset(winter_day_image_paths, feature_detector, num_dict_features, trained_model)
winter_night_accuracy = evaluate_model_on_subset(winter_night_image_paths, feature_detector, num_dict_features, trained_model)
spring_day_accuracy = evaluate_model_on_subset(spring_day_image_paths, feature_detector, num_dict_features, trained_model)
spring_night_accuracy = evaluate_model_on_subset(spring_night_image_paths, feature_detector, num_dict_features, trained_model)
 
# Print results
print(f"Autumn Accuracy: {autumn_accuracy}")
print(f"Winter Accuracy: {winter_accuracy}")
print(f"Spring Accuracy: {spring_accuracy}")
 
print(f"Autumn Day Accuracy: {autumn_day_accuracy}")
print(f"Autumn Night Accuracy: {autumn_night_accuracy}")
print(f"Winter Day Accuracy: {winter_day_accuracy}")
print(f"Winter Night Accuracy: {winter_night_accuracy}")
print(f"Spring Day Accuracy: {spring_day_accuracy}")
print(f"Spring Night Accuracy: {spring_night_accuracy}")

data = [
    ("Autumn", autumn_accuracy),
    ("Winter", winter_accuracy),
    ("Spring", spring_accuracy),
    ("Autumn Day", autumn_day_accuracy),
    ("Autumn Night", autumn_night_accuracy),
    ("Winter Day", winter_day_accuracy),
    ("Winter Night", winter_night_accuracy),
    ("Spring Day", spring_day_accuracy),
    ("Spring Night", spring_night_accuracy),
]

csv_file_path = "accuracy_data.csv"

# Write data to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers
    writer.writerow(["PORA DNIA I ROKU", "DOKŁADNOSC"])
    
    # Write data
    writer.writerows(data)

print(f"Data has been saved to {csv_file_path}")