import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ----------- Feature Extraction -----------
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs, axis=1)
        return mfccs_scaled
    except Exception as e:
        print("Error in file:", file_path, "|", e)
        return None

# ----------- Dataset Path -----------
dataset_path = "C:\\Users\\Kanchan\\OneDrive\\Desktop\\machine learning\\dataset"

features = []
labels = []

# ----------- Load Dataset -----------
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion = os.path.basename(root)

            data = extract_features(file_path)

            if data is not None and len(data) == 40:
                features.append(data)
                labels.append(emotion)

X = np.array(features)
y = np.array(labels)

if len(X) == 0:
    raise ValueError("❌ Dataset empty hai ya path galat hai")

print("✅ Total samples:", len(X))

# ----------- Encode Labels -----------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ----------- Train Test Split -----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ----------- Model (Random Forest) -----------
model = RandomForestClassifier(n_estimators=200, random_state=42)

# ----------- Training -----------
model.fit(X_train, y_train)

# ----------- Prediction -----------
y_pred = model.predict(X_test)

# ----------- Evaluation -----------
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Accuracy:", accuracy)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
