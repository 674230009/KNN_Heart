from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกข้อมูลดอกไม้")
#st.image("./img/kairung022.jpg")
col1, col2, col3 = st.columns(3)

with col1:
   st.header("เป็นโรคหัวใจ")
   st.image("./img/Heard1.jpg")

with col2:
   st.header("ไม่เป็นโรคหัวใจ")
   st.image("./img/Heard2.png")

with col2:
   st.header("Profile")
   st.image("./img/Warapon.jpg")


html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['Age'].mean()
dt2 = dt['Sex'].mean()
dt3 = dt['ChestPainType'].mean()
dt4 = dt['MaxHR'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["age", "sex", "cp", "hr"], columns=["ค่าเฉลี่ย"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   # st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

#รับค่าอินพุตให้ครบ 11 คอลัมน์ตามไฟล์ Heart3csv
col1, col2 = st.coiumns(2)

age = st.slider("กรุณาเลือกข้อมูล Age")
sex = st.slider("กรุณาเลือกข้อมูล Sex")
cp = st.slider("กรุณาเลือกข้อมูล ChestPainType")
hr = st.slider("กรุณาเลือกข้อมูล MaxHR")

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris.csv") 

   X = dt.drop('variety', axis=1)
   y = dt.variety   
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]]) #ข้อมูลใหม่สำหรับทำนาย

   st.write(Knn_model.predict(x_input))   
   out=Knn_model.predict(x_input)

   if out[0] == 'Setosa':
    st.image("./img/Heard1.jpg")
   else:
    st.image("./img/Heard2.jpg")
else:
    st.write("ไม่ทำนาย")
