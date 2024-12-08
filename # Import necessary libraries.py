# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\Rain\Desktop\옾소\OpenProject\diabetes_data.xlsx"  # 수정된 경로
data = pd.read_excel(file_path)  # 엑셀 파일 읽기
print(data.head())  # 데이터 확인

# 1. Data Preprocessing
# Select relevant columns, including 'physical_activity'
data = data[['diet', 'medication_adherence', 'physical_activity', 'blood_glucose', 'risk_score']]

# Check for missing values
print("Missing values per column:")
print(data.isnull().sum())

# Verify the data structure
print("Data head:")
print(data.head())

# 2. Groupby Analysis
grouped = data.groupby(['diet', 'medication_adherence', 'physical_activity'])[['blood_glucose', 'risk_score']].agg(['mean', 'std', 'count'])
print("Groupby analysis results:")
print(grouped)

# 3. Data Visualization

# 3.1 Violin Plot: Distribution of Blood Glucose by Diet and Medication Adherence
plt.figure(figsize=(10, 6))
sns.violinplot(x='diet', y='blood_glucose', hue='medication_adherence', data=data, split=True, palette='coolwarm')
plt.title('Blood Glucose Distribution by Diet and Medication Adherence')
plt.xlabel('Diet (0: Non-Compliant, 1: Compliant)')
plt.ylabel('Blood Glucose')
plt.legend(title='Medication Adherence (0/1)')
plt.show()

# 3.2 Pair Plot: Relationships Among Variables
sns.pairplot(data, hue='diet', palette='viridis', diag_kind='kde', corner=True)
plt.suptitle("Pairwise Relationships Among Variables", y=1.02)
plt.show()

# 3.3 Facet Grid: Blood Glucose vs Physical Activity by Diet
facet = sns.FacetGrid(data, col="diet", hue="medication_adherence", palette="coolwarm", height=4, aspect=1.5)
facet.map(sns.scatterplot, "physical_activity", "blood_glucose", alpha=0.7)
facet.add_legend()
facet.set_axis_labels("Physical Activity (Hours)", "Blood Glucose")
facet.set_titles("Diet: {col_name}")
plt.subplots_adjust(top=0.85)
plt.suptitle("Blood Glucose vs Physical Activity by Diet and Medication Adherence", y=1.02)
plt.show()

# 4. Machine Learning: Linear Regression

# Prepare data for modeling
X = data[['diet', 'medication_adherence', 'physical_activity']]
y = data['blood_glucose']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

# 5. Save Processed Data for Reporting
processed_file_path = '/mnt/data/processed_diabetes_subset_with_activity.csv'
data.to_csv(processed_file_path, index=False)

print(f"Processed data saved to {processed_file_path}")
