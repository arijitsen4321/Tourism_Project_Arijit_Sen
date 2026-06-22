import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ==========================================================

# CONFIG

# ==========================================================

REPO_ID = "arijitsen4321/Tourism_Project_Arijit_Sen"

# ==========================================================

# LOAD MODEL

# ==========================================================

model_path = hf_hub_download(
repo_id=REPO_ID,
filename="Best_Tourism_predict_model_v1.joblib",
repo_type="model"
)

model = joblib.load(model_path)

# ==========================================================

# LOAD TRAINING COLUMNS

# ==========================================================

xtrain_path = hf_hub_download(
repo_id=REPO_ID,
filename="Xtrain.csv",
repo_type="dataset"
)

Xtrain = pd.read_csv(xtrain_path)

if "Unnamed: 0" in Xtrain.columns:
Xtrain.drop(columns=["Unnamed: 0"], inplace=True)

TRAIN_COLUMNS = Xtrain.columns.tolist()

# ==========================================================

# UI

# ==========================================================

st.title("Tourism Product Acceptance Prediction")

st.write(
"Predict whether a customer is likely to purchase a tourism package."
)

col1, col2 = st.columns(2)

with col1:

```
Age = st.number_input("Age", 18, 100, 35)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch",
    min_value=0,
    value=15
)

NumberOfPersonVisiting = st.number_input(
    "Number Of Person Visiting",
    min_value=1,
    value=2
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    min_value=0,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

NumberOfTrips = st.number_input(
    "Number Of Trips",
    min_value=0,
    value=2
)

Occupation = st.selectbox(
    "Occupation",
    [
        "Large Business",
        "Salaried",
        "Small Business"
    ]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Deluxe",
        "King",
        "Standard",
        "Super Deluxe"
    ]
)
```

with col2:

```
Passport = st.selectbox(
    "Passport",
    [0, 1]
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0,
    value=0
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=30000
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    [
        "Married",
        "Single",
        "Unmarried"
    ]
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "VP"
    ]
)

TypeofContact = st.selectbox(
    "Type Of Contact",
    [
        "Self Enquiry",
        "Other"
    ]
)
```

# ==========================================================

# PREDICTION

# ==========================================================

if st.button("Predict"):

```
input_df = pd.DataFrame(
    [[0] * len(TRAIN_COLUMNS)],
    columns=TRAIN_COLUMNS
)

# Numerical columns

input_df.loc[0, "Age"] = Age
input_df.loc[0, "CityTier"] = CityTier
input_df.loc[0, "DurationOfPitch"] = DurationOfPitch
input_df.loc[0, "NumberOfPersonVisiting"] = NumberOfPersonVisiting
input_df.loc[0, "NumberOfFollowups"] = NumberOfFollowups
input_df.loc[0, "PreferredPropertyStar"] = PreferredPropertyStar
input_df.loc[0, "NumberOfTrips"] = NumberOfTrips
input_df.loc[0, "Passport"] = Passport
input_df.loc[0, "PitchSatisfactionScore"] = PitchSatisfactionScore
input_df.loc[0, "OwnCar"] = OwnCar
input_df.loc[0, "NumberOfChildrenVisiting"] = NumberOfChildrenVisiting
input_df.loc[0, "MonthlyIncome"] = MonthlyIncome

# Helper

def set_dummy(col_name):
    if col_name in input_df.columns:
        input_df.loc[0, col_name] = 1

# Categorical columns

set_dummy(f"Gender_{Gender}")
set_dummy(f"MaritalStatus_{MaritalStatus}")
set_dummy(f"Occupation_{Occupation}")
set_dummy(f"ProductPitched_{ProductPitched}")
set_dummy(f"Designation_{Designation}")

if TypeofContact == "Self Enquiry":
    set_dummy("TypeofContact_Self Enquiry")

# Ensure exact order

input_df = input_df.reindex(
    columns=TRAIN_COLUMNS,
    fill_value=0
)

st.write("Input Shape:", input_df.shape)

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")

if prediction == 1:
    st.success(
        f"Customer is likely to purchase the tourism package. Probability: {probability:.2%}"
    )
else:
    st.error(
        f"Customer is unlikely to purchase the tourism package. Probability: {probability:.2%}"
    )
```
