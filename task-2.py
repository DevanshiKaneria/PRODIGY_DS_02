import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths — change these if needed
train_file = "train.csv"
test_file = "test.csv"
submission_file = "gender_submission.csv"

# Load the data
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
submission_df = pd.read_csv(submission_file)

# Add a column to distinguish train/test
train_df['Dataset'] = 'Train'
test_df['Dataset'] = 'Test'

# In test data, Survived is missing — fill with NaN for now
test_df['Survived'] = pd.NA

# Combine train & test for EDA
df = pd.concat([train_df, test_df], ignore_index=True)


# Basic Info

print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())


# Data Cleaning

# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill Fare with median (in test set)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop Cabin (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Verify
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Exploratory Data Analysis

# Only use Train data when analyzing survival
train_only = df[df['Dataset'] == 'Train']

# Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=train_only, palette='Set2')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Survival by Sex
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=train_only, palette='Set1')
plt.title("Survival by Sex")
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(train_only['Age'], bins=30, kde=True, color='blue')
plt.title("Age Distribution (Train)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Age vs. Survival
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Age', data=train_only, palette='cool')
plt.title("Age vs. Survival")
plt.show()

# Passenger Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=train_only, palette='Set3')
plt.title("Passenger Class Distribution (Train)")
plt.show()

# Pclass vs. Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=train_only, palette='muted')
plt.title("Survival by Passenger Class")
plt.show()

# Fare distribution
plt.figure(figsize=(8,5))
sns.histplot(train_only['Fare'], bins=40, kde=True, color='green')
plt.title("Fare Distribution (Train)")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(train_only.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Train)")
plt.show()


# Insights

print("\nSome observations:")
print("- Women had much higher survival rate than men.")
print("- Passengers in 1st class survived more than 2nd/3rd class.")
print("- Younger passengers slightly more likely to survive.")
print("- Higher fare correlates with higher survival.")

