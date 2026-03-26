import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Setup paths
DATA_FILE = r"D:\archive\Project_Dataset\extracted_features.npz"

def main():
    print("Loading extracted features...")
    # Load the .npz file we created in the last step
    data = np.load(DATA_FILE)
    features = data['features']
    labels = data['labels']
    
    # 2. Split the data
    # 80% for training the ML model, 20% for testing it on unseen data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"Training on {len(X_train)} images, testing on {len(X_test)} unseen images.\n")

    # 3. Scale the data
    # SVMs work much better when all 512 numbers are scaled to a standard range
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the Support Vector Machine
    print("Training the SVM Classifier... (This takes a few seconds)")
    # 'rbf' is a great kernel for complex, non-linear image features
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # 5. Make Predictions
    print("Testing the model...\n")
    predictions = svm_model.predict(X_test_scaled)

    # 6. Print the Final Report
    # These must match the alphabetical order of the folders we created
    class_names = [
        'Hand_Broken', 'Hand_Healthy', 
        'Shoulder_Broken', 'Shoulder_Healthy', 
        'Wrist_Broken', 'Wrist_Healthy'
    ]

    print("=========================================")
    print("         FINAL MODEL RESULTS             ")
    print("=========================================")
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print("Detailed Breakdown:")
    print(classification_report(y_test, predictions, target_names=class_names))

if __name__ == "__main__":
    main()