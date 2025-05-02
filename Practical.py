import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
train_path = r"C:\Users\Shreyash Musmade\Desktop\Practical\AICS\AICS_Prac-4\train.csv"
test_path = r"C:\Users\Shreyash Musmade\Desktop\Practical\AICS\AICS_Prac-4\test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

# Combine Train and Test for Consistent Feature Engineering
data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# 1. Handle Missing Values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# 2. Drop Columns with Excessive Missing Data
data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# 3. Encode Categorical Features
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

# 4. Feature Scaling
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# 5. Create New Features
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

print("Feature Engineering Completed.")
print(data.head())
# Split back to train and test
train_final = data[:len(train_data)]
test_final = data[len(train_data):]

print("Final Train Shape:", train_final.shape)
print("Final Test Shape:", test_final.shape)
