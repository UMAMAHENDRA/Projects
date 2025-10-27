import pandas as pd
import streamlit as st
import pickle 

with open("used_phone.pkl",'rb') as f:
    model = pickle.load(f)


df = pd.read_csv('used_phone.csv')

brand_model_map = df.groupby('brand')['model'].unique().to_dict()

brand_list = list(brand_model_map.keys())


st.title("Used Phone Price Prediction")
st.sidebar.header("Enter Phone Details")

selected_brand = st.sidebar.selectbox("Brand",brand_list)
selected_model = st.sidebar.selectbox("Model",brand_model_map[selected_brand])
selected_ram = st.sidebar.selectbox("Ram",sorted(df['ram_gb'].unique()))
selected_storage = st.sidebar.selectbox("Storage",sorted(df['storage_gb'].unique()))
selected_condition = st.sidebar.selectbox("Condintion of Phone",df['condition'].unique())
selected_battery = st.sidebar.slider("Battery Health",50,100,80)
selected_age = st.sidebar.slider("Used for",0,5,1)
selected_price = st.sidebar.number_input("Original Price (INR)",3000,200000,15000)


from sklearn.preprocessing import LabelEncoder

# Fit encoders
le_brand = LabelEncoder()
le_model = LabelEncoder()
le_condition = LabelEncoder()

df['brand'] = le_brand.fit_transform(df['brand'])
df['model'] = le_model.fit_transform(df['model'])
df['condition'] = le_condition.fit_transform(df['condition'])

# Transform user inputs
brand_encoded = le_brand.transform([selected_brand])[0]
model_encoded = le_model.transform([selected_model])[0]
condition_encoded = le_condition.transform([selected_condition])[0]

input_data = pd.DataFrame({
    "brand" : [brand_encoded],
    "model" : [model_encoded],
    "ram_gb" : [selected_ram],
    "storage_gb" : [selected_storage],
    "condition" : [condition_encoded],
    "battery_health" : [selected_battery],
    "age_years" : [selected_age],
    "original_price" : [selected_price]
})

if st.sidebar.button("Predict"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Used Phone Price : {int(predicted_price):,}")

