import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle
import streamlit as st

# Load the trained model
model = keras.models.load_model("model.h5")
# Load the encoders and scaler
with open("level_encoder_gender.pkl","rb") as f:
    level_encoder_gender = pickle.load(f)

with open("one_hot_encoder_geography.pkl","rb") as f:
    one_hot_encoder_geography = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

## streamlit app 
st.title("Customer Churn prediction")

#taking input from the user
credict_score = st.slider("Enter credit score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Select geography", one_hot_encoder_geography.categories_[0])
gender = st.selectbox("Select gender", level_encoder_gender.classes_)
age = st.number_input("Enter age", min_value=0, max_value=100, value=30)
tenure = st.slider("Enter tenure", min_value=0, max_value=10, value=1)
balance = st.number_input("Enter balance", min_value=0, value=0)
num_of_products = st.slider("Enter number of products", min_value=1, max_value=5, value=1)
has_cr_card = st.selectbox("Has credit card?", [0, 1])
is_active_member = st.selectbox("Is active member?", [0, 1])
estimated_salary = st.number_input("Enter estimated salary", min_value=0, value=0)

#prepare the data set which we got from the user input for the prediction 
input_data = pd.DataFrame({
    "CreditScore": [credict_score],
    "Gender": [level_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

#one hot encoding for the geography column
geo_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = one_hot_encoder_geography.get_feature_names_out())

#combine the encoded and original data frame 
input_df = pd.concat([input_data.reset_index(drop = True), geo_encoded_df],axis = 1)

#scaling the recieved data from the user 
input_df_scaled = scaler.transform(input_df)

#now lets do the prediction
prediction = model.predict(input_df_scaled)
prediction_prob = prediction[0][0]

#analysing the prediction
if prediction_prob > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_prob:.2f}")
else:
    st.write(f"The customer is not likely to churn with a probability of {1 - prediction_prob:.2f}")