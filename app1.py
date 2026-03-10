import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Employee Dataset Machine Learning App")

# ---------------- STEP 1 : LOAD DATASET ----------------
st.header("1. Load Dataset")

df = pd.read_csv("Employee_Dataset.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

st.write("Shape of Dataset:", df.shape)

# ---------------- STEP 2 : UNDERSTAND & CLEAN DATA ----------------
st.header("2. Understand Data and Data Cleaning")

st.subheader("Dataset Info")
st.write(df.dtypes)

st.subheader("Missing Values")
st.write(df.isnull().sum())

# Convert columns to numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
df['experience_years'] = pd.to_numeric(df['experience_years'], errors='coerce')
df['performance_rating'] = pd.to_numeric(df['performance_rating'], errors='coerce')

# Fill missing values
df['age'] = df['age'].fillna(df['age'].mean())
df['salary'] = df['salary'].fillna(df['salary'].mean())
df['experience_years'] = df['experience_years'].fillna(df['experience_years'].mean())
df['performance_rating'] = df['performance_rating'].fillna(df['performance_rating'].median())

# Drop unnecessary columns
df = df.drop(columns=['employee_id','joining_date','last_promotion_date'], errors='ignore')

# Convert categorical columns
df = pd.get_dummies(df, columns=['department','designation'], drop_first=True)

# Convert boolean column
if 'is_active' in df.columns:
    df['is_active'] = df['is_active'].astype(str).map({'True':1,'False':0})

# Fill any remaining NaN values
df = df.fillna(0)

st.write("Cleaned Dataset")
st.dataframe(df.head())

# ---------------- STEP 3 : INPUT OUTPUT & VISUALIZATION ----------------
st.header("3. Find Input and Output Variables & Visualization")

X = df.drop("salary", axis=1)
y = df["salary"]

st.write("Input Features")
st.write(X.columns)

st.write("Output Variable: Salary")

# Visualization
st.subheader("Salary Distribution")

fig, ax = plt.subplots()
ax.hist(y, bins=20)
ax.set_title("Salary Distribution")
st.pyplot(fig)

# ---------------- STEP 4 : TRAIN TEST SPLIT ----------------
st.header("4. Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("Training Data Shape:", X_train.shape)
st.write("Testing Data Shape:", X_test.shape)

# ---------------- STEP 5 : CREATE MODEL & TRAIN ----------------
st.header("5. Create Model and Train")

model = LinearRegression()
model.fit(X_train, y_train)

st.success("Model Training Completed")

# ---------------- STEP 6 : TESTING ----------------
st.header("6. Testing")

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

st.write("Model R2 Score:", score)

# ---------------- STEP 7 : PREDICTION ----------------
st.header("7. Salary Prediction")

age = st.number_input("Enter Age", 18, 60)
experience = st.number_input("Experience Years", 0, 40)
rating = st.slider("Performance Rating", 1, 5)

if st.button("Predict Salary"):

    # Create input dataframe with same columns
    input_data = pd.DataFrame(columns=X.columns)

    # Set default values
    input_data.loc[0] = 0

    # Fill user inputs
    if 'age' in input_data.columns:
        input_data['age'] = age

    if 'experience_years' in input_data.columns:
        input_data['experience_years'] = experience

    if 'performance_rating' in input_data.columns:
        input_data['performance_rating'] = rating

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: {prediction[0]:.2f}")