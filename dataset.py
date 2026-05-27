import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Enable interactive mode
plt.ion()

# Load Dataset
df = pd.read_csv(r"C:\Users\shwet\Downloads\titanic.csv")

# 1. Display First 5 Rows
print("\nFIRST 5 ROWS OF DATASET")
print(df.head())

# 2. Dataset Information
print("\nDATASET INFORMATION")
df.info()

# 3. Statistical Summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# 4. Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\nMISSING VALUES AFTER HANDLING")
print(df.isnull().sum())

print(df.isnull().sum())
# 5. Column Names
print("\nCOLUMN NAMES")
print(df.columns)

# 6. Duplicate Rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# 7. Hypothesis Testing

# Survival Rate by Gender
print("\nSURVIVAL RATE BY GENDER")
print(df.groupby('Sex')['Survived'].mean())

# Survival Rate by Passenger Class
print("\nSURVIVAL RATE BY PASSENGER CLASS")
print(df.groupby('Pclass')['Survived'].mean())

# 8. Visualization

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("graph1.png")
plt.show()

# Gender vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Gender vs Survival")
plt.savefig("graph2.png")
plt.show()

# Passenger Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.savefig("graph3.png")
plt.show()

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'].dropna(), bins=30)
plt.title("Age Distribution")
plt.savefig("graph4.png")
plt.show()

# Fare Outliers
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.savefig("graph5.png")
plt.show()


