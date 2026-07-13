import pandas as pd
import numpy as np
import datetime 
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st

def main():
    html_temp = """<h1>Salary Prediction</h1>"""
    
    model = GradientBoostingRegressor()
    model.load_model("xgb_model.pkl")

    st.markdown(html_temp, unsafe_allow_html = True)

    st.markdown("This app will help you predict your Salary based on your experience, education, and other factors. Please enter the required details below.")

    p1 = st.number_input("Please enter Age", 17, 80, step=1)

    s1 = st.selectbox("Please enter your Gender", ["Female", "Male"])
    if s1=="Female":
        p2 = 0
    elif s1=="Male":
        p2 = 1

    s2 = st.selectbox("Please select education level", ["Diploma", "Bachelor", "Master", "PhD"])
    if s2=="Diploma":
        p3 = 0
    elif s2=="Bachelor":
        p3 = 1
    elif s2=="Master":
        p3 = 2
    elif s2=="PhD":
        p3 = 3
        p3 = 0


    p4 = st.number_input("Please enter experience years", 0, 70.0, step=1.0)
    
    s3 = st.selectbox("Please select your Department", [ 'Operations','IT','Finance','Sales','HR','Marketing'])
    if s3=="Operations":
        p5 = 0
    elif s3=="IT":
        p5 = 1
    elif s3=="Finance":
        p5 = 2
    elif s3=="Sales":
        p5 = 3
    elif s3=="HR":
        p5 = 4
    elif s3=="Marketing":
        p5 = 5
    

    s4 = st.selectbox("Please select your job type", ['Junior', 'Mid', 'Senior', 'Lead', 'Manager'])
    if s4=="Junior":        
        p6 = 0
    elif s4=="Mid":
        p6 = 1
    elif s4=="Senior":
        p6 = 2
    elif s4=="Lead":
        p6 = 3
    elif s4=="Manager":
        p6 = 4
    

    p7 = st.slider("Give your Performance rating",0,5,step=1)

    p8 = st.slider("Give your Certifications",0,10,step=1)

    p9 = st.number_input("Please enter Overtime Hours", 0, 70.0, step=1.0)

    s5 = st.selectbox("Please select remote or not", ["Yes", "No"])
    if s5=="Yes":
        p10 = 1
    elif s5=="No":
        p10 = 0

    s6 = st.selectbox("Please select City", ['Hyderabad','Mumbai','Pune','Chennai','Bangalore','Delhi'])
    if s6=="Hyderabad":
        p11 = 0
    elif s6=="Mumbai":
        p11 = 1
    elif s6=="Pune":
        p11 = 2
    elif s6=="Chennai":
        p11 = 3
    elif s6=="Bangalore":
        p11 = 4
    elif s6=="Delhi":
        p11 = 5

    p12 = st.number_input("Please enter your Company Tenure", 0.0, 15.0, step=1)
    
    p13 = st.number_input("Please enter Project completed", 0.0, 40.0, step=1)
    
    p14 = st.number_input("Please enter your Skill rating", 50.0, 200.0, step=1)


    data_new = pd.DataFrame({
    'Age' :p1,
    'Gender' : p2,
    'Education' :p3,
    'Experience_Years':p4,
    'Department':p5,
    'Job_Level':p6,
    'Performance_Rating':p7,
    'Certifications':p8,
    'Overtime_Hours':p9,
    'Remote_Work':p10,
    'City':p11,
    "Company_Tenure":p12,
    'Projects_Completed':p13,
    'Skill_Score':p14
    
},index=[0])

    if st.button("Predict"):
        pred = model.predict(data_new)
        st.success("Your predicted salary is: {:.2f} Lakhs".format(pred[0]))



if __name__=='__main__':
    main()