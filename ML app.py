import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st
heart_data = pd.read_csv('heart_disease_data.csv')



X = heart_data.drop('target',axis = 1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


#website
st.title("Heart Disease Prediction Model")
input_text = st.text_input('Provide comma seperated values')
img = Image.open("heart_image.jpg") 
 
# Adjust the path to your image file

st.image(img, width=150)
try:
    df = pd.DataFrame([float(i) for i in input_text.split(',')]).T
    df.columns = X.columns  # Ensure columns match training data
    prediction = model.predict(df)
    if prediction[0] == 0:
        st.write("The person does not have heart disease")
    else:
        st.write("The person has heart disease")
except Exception:
    st.write("This person have heart disease")

    st.subheader("About the Model")
    st.write(heart_data)
    st.subheader("Model performance on Training Data")
    st.write(training_data_accuracy)
    st.subheader("Model performance on Test Data")
    st.write(test_data_accuracy)





    
