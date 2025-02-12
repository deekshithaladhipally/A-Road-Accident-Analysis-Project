import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('accidents_india.csv')

# Preprocessing (filling missing values, encoding)
df.dropna(inplace=True)
df['Severity'] = LabelEncoder().fit_transform(df['Accident_Severity'])
df['Day'] = LabelEncoder().fit_transform(df['Day_of_Week'])
df['Light'] = LabelEncoder().fit_transform(df['Light_Conditions'])
df.drop(['Accident_Severity', 'Day_of_Week', 'Light_Conditions'], axis=1, inplace=True)

# Define features and target
X = df.drop(['Pedestrian_Crossing', 'Special_Conditions_at_Site', 'Severity'], axis=1)
y = df['Severity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for oversampling minority class
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Train Decision Tree with class weights
dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt.fit(X_train_sm, y_train_sm)
y_pred_dt = dt.predict(X_test)

# Train Random Forest with class weights
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train_sm, y_train_sm)
y_pred_rf = rf.predict(X_test)

# Display results
print("Decision Tree - Classification Report:")
print(classification_report(y_test, y_pred_dt))

print("Random Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()