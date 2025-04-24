# train/train_model.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.outlier_handler import OutlierHandler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# === 1. تحميل البيانات ===
df = pd.read_csv("data/filled_data (1).csv")

# حذف العمود غير الضروري
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# تأكد أن العمود الهدف نوعه صحيح
df['Heat stroke'] = df['Heat stroke'].astype(int)

# === 2. اختيار الأعمدة ===
features = ['Environmental temperature (C)', 'Heart / Pulse rate (b/min)', 'Sweating', 'Patient temperature']
target = 'Heat stroke'

X = df[features]
y = df[target]

# === 3. معالجة القيم الشاذة ===
outlier_handler = OutlierHandler()
X[['Environmental temperature (C)', 'Heart / Pulse rate (b/min)']] = outlier_handler.fit_transform(
    X[['Environmental temperature (C)', 'Heart / Pulse rate (b/min)']]
)

# === 4. القياس ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 5. تقسيم البيانات ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 6. تدريب نموذج SVM ===
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# === 7. التقييم ===
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# === 8. حفظ الكائنات ===
with open("models/Heat.Stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/Scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/outlier_handler.pkl", "wb") as f:
    pickle.dump(outlier_handler, f)

print("\n✅ Model and preprocessing tools saved successfully.")