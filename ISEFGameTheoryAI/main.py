import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def label_infinite_nim(heaps):
    nz_count = np.count_nonzero(heaps)
    total_heaps = len(heaps)
    if nz_count > total_heaps * 0.6:  
        return 0
    return 1 if np.bitwise_xor.reduce(heaps[:nz_count]) != 0 else 0

def generate_dataset(num_samples=50000, num_heaps=100, max_heap_size=30):
    data = []
    for _ in range(num_samples):
        support_size = np.random.geometric(0.01)
        support_size = min(support_size, num_heaps)
        heaps = np.zeros(num_heaps, dtype=int)
        if np.random.rand() < 0.5:
            heaps = np.random.randint(1, max_heap_size+1, size=num_heaps)
        else:
            idxs = np.random.choice(num_heaps, size=support_size, replace=False)
            heaps[idxs] = np.random.randint(1, max_heap_size+1, size=support_size)
        label = label_infinite_nim(heaps)
        data.append((heaps.tolist(), label))
    return pd.DataFrame(data, columns=['heaps', 'win_label'])

def add_features(df):
    heaps = np.vstack(df['heaps'])
    features = pd.DataFrame()
    features['xor'] = np.bitwise_xor.reduce(heaps, axis=1)
    features['nz_count'] = (heaps > 0).sum(axis=1)
    features['max_heap'] = heaps.max(axis=1)
    features['sum_heaps'] = heaps.sum(axis=1)
    features['parity_sum'] = features['sum_heaps'] % 2
    B = 8
    bit_planes = []
    for b in range(B):
        bit_planes.append(((heaps >> b) & 1).sum(axis=1))
    bit_planes = np.vstack(bit_planes).T
    bit_df = pd.DataFrame(bit_planes, columns=[f'bit_sum_{b}' for b in range(B)])
    return pd.concat([features, bit_df], axis=1)

df = generate_dataset()
X = add_features(df)
y = df['win_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

smoteenn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning ------------------------------------

param_grid = {
    'n_estimators': [100, 300, 700],
    'max_depth': [None, 10, 20],
}

base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)

grid = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

print("Running Grid Search...")
grid.fit(X_train_resampled, y_train_resampled)

print("Best parameters found:", grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)

model = grid.best_estimator_

# evaluation ------------------------------

# model = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=42)
# model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test_scaled)
print("="*40)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred, target_names=['Losing (0)', 'Winning (1)'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nDetailed classification report:\n", report_df)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, y_pred):.4f}")

metrics_to_plot = ['precision', 'recall', 'f1-score']
sns.heatmap(report_df.loc[['Losing (0)', 'Winning (1)'], metrics_to_plot], annot=True, cmap='Blues')
plt.title('Classification Metrics per Class')
plt.show()

importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title('Random Forest Feature Importance')
plt.show()

with open('rf_model_lipparini.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler_lipparini.pkl', 'wb') as f:
    pickle.dump(scaler, f)
