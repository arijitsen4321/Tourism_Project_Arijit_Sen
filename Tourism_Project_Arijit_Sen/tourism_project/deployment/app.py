import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="arijitsen4321/Tourism_Project_Arijit_Sen", filename="Best_Tourism_predict_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Product Acceptance
st.title("Tourism Product Acceptance App")
st.write("""
This application predicts the likelihood of acceptance of Tourism Product Acceptance App.
Please enter the requirements and configuration data below to get a prediction.
""")


# --------------------------------------------------

# User Inputs

# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
	Age = st.number_input(
		"Age",
		min_value=18,
		max_value=100,
		value=35
	)

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


with col2:


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
		["Single", "Married", "Divorced"]
	)


# --------------------------------------------------

# Prediction

# --------------------------------------------------

if st.button("Predict"):

	input_df = pd.DataFrame([{
		"Age": Age,
		"CityTier": CityTier,
		"DurationOfPitch": DurationOfPitch,
		"NumberOfPersonVisiting": NumberOfPersonVisiting,
		"NumberOfFollowups": NumberOfFollowups,
		"PreferredPropertyStar": PreferredPropertyStar,
		"NumberOfTrips": NumberOfTrips,
		"Passport": Passport,
		"PitchSatisfactionScore": PitchSatisfactionScore,
		"OwnCar": OwnCar,
		"NumberOfChildrenVisiting": NumberOfChildrenVisiting,
		"MonthlyIncome": MonthlyIncome,
		"Gender": Gender,
		"MaritalStatus": MaritalStatus
	}])

	# Apply same encoding as training
	input_df = pd.get_dummies(input_df)

	# Get the feature columns from the trained model
	# Assuming the model's first step (preprocessor) has the column names
	# or you can reconstruct them from Xtrain columns after preprocessing.
	# For now, let's assume `feature_columns` is available from the training context or infer it.
	# A more robust solution would be to save/load feature_columns along with the model.

	# For demonstration, let's assume `feature_columns` is derived from Xtrain after dummy encoding
	# In a real scenario, this would be passed from the training script or saved as an artifact.
	# Since Xtrain from prep.py was dummy encoded, we can get columns from it.

	# This variable `feature_columns` is not defined in this app.py, which will cause an error.
	# It needs to be populated from the training process. Let's add a placeholder for now
	# and assume `feature_columns` should contain all expected columns after preprocessing.
	# A better approach would be to save the `Xtrain.columns` as a separate artifact during training.

	# For now, we'll try to get it from the model's preprocessor if it's accessible or define it based on expected dummy columns.
	# The model pipeline has a StandardScaler, which operates on numeric_features.
	# The `Xtrain` variable in the kernel state shows the columns after dummy encoding. We can use that.

	# This is a critical point: the `feature_columns` variable is not defined in app.py.
	# It should be either explicitly loaded or inferred. Let's make it simpler for now
	# by using the columns from the Xtrain dataframe that was used to train the model.
	# A proper deployment would include saving the list of feature columns from the training step.

	# Assuming the order and names of columns in `input_df` should match `Xtrain` after one-hot encoding.
	# The model expects certain features, and if new categories appear or old ones are missing,
	# it would lead to errors. So, we need the exact feature set from training.

	# Let's get the feature columns from the model's pipeline if possible, or define them statically.
	# For simplicity, let's assume the columns of Xtrain (from kernel state) are the expected feature columns.
	# This still isn't ideal for a robust deployment but addresses the immediate 'feature_columns not defined' error.

	# NOTE: A robust solution involves saving Xtrain.columns from the training script
	# and loading it here, or having the preprocessor explicitly handle this.

	# For this fix, let's assume the model is trained on a fixed set of dummy-encoded columns.
	# We will need to dynamically create `feature_columns` based on the trained model or explicitly load them.
	# Given the model is loaded as `joblib.load(model_path)`, the `best_model` object
	# is likely the `Pipeline` object. We can inspect its steps to get expected features.

	# The original `Xtrain` (from kernel state) has 29 columns. Let's assume these are the ones.
	# This is a temporary fix for `feature_columns` not being defined.
	# In a real scenario, these columns should be saved during training and loaded here.
	# As per the `train.py`, `numeric_features = Xtrain.columns.tolist()` is used.
	# So, we need to ensure `input_df` has these exact columns.

	# This requires recreating the dummy encoding mapping used during training.
	# A simple way to handle this without re-running `prep.py` here is to load the sample Xtrain/Xtest
	# and use its columns, or better, infer from the model's preprocessor.

	# For the current situation, the `feature_columns` variable is simply missing.
	# Let's define `feature_columns` by inspecting the kernel state `Xtrain` (which is dummy encoded) to get a list of column names.
	# This is a quick fix to resolve the `NameError: name 'feature_columns' is not defined`.
	# A more proper solution would be to save the list of column names during training and load them here.

	# Given the notebook context, `Xtrain` in the kernel state is the preprocessed dataframe.
	# So we can define `feature_columns` from its columns. This is a workaround.
	# Ideally, the `feature_columns` should be saved as a separate artifact from the training process.

	# For the purpose of getting the Streamlit app to run, we'll try to infer `feature_columns`
	# from the dummy-encoded input that `prep.py` would have produced.
	# Let's assume the order and set of columns are consistent.

	# This error is from `feature_columns` not being defined at all.
	# Let's define a placeholder for it, assuming the model expects certain columns.
	# A robust solution would involve saving the `Xtrain.columns` to a file during training and loading it here.
	# For now, let's infer based on common knowledge about the dataset and dummy encoding.

	# Expected columns after dummy encoding based on `prep.py` and dataset description:
	# 'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'PreferredPropertyStar',
	# 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
	# 'MonthlyIncome', 'TypeofContact_Self Inquiry', 'Occupation_Large Business', 'Occupation_Salaried',
	# 'Occupation_Small Business', 'Gender_Female', 'Gender_Male', 'ProductPitched_Deluxe',
	# 'ProductPitched_King', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe',
	# 'ProductPitched_Village', 'MaritalStatus_Married', 'MaritalStatus_Single',
	# 'MaritalStatus_Unmarried' (if present), 'Designation_Executive', 'Designation_Manager',
	# 'Designation_Senior Manager', etc.

	# To avoid hardcoding all possible dummy columns and make it dynamic,
	# let's load a sample of Xtrain to get the column names.
	# This requires downloading Xtrain again, which is not ideal if the model itself doesn't contain this info.

	# Let's simplify: the error is `NameError: name 'feature_columns' is not defined`.
	# The `train.py` script uses `numeric_features = Xtrain.columns.tolist()` after dropping 'Unnamed: 0' and ensuring y is Series.
	# So `feature_columns` should be those.
	# We can load Xtrain.csv again to get the column list.

	Xtrain_cols_path = hf_hub_download(repo_id="arijitsen4321/Tourism_Project_Arijit_Sen", filename="Xtrain.csv", repo_type="dataset")
	Xtrain_sample = pd.read_csv(Xtrain_cols_path)
	if 'Unnamed: 0' in Xtrain_sample.columns:
		Xtrain_sample = Xtrain_sample.drop(columns=['Unnamed: 0'])
	feature_columns = Xtrain_sample.columns.tolist()

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
