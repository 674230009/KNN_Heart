from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("Heart Disease")
#st.image("./img/kairung022.jpg")
col1, col2, col3 = st.columns(3)

with col1:
   st.header("heart disease")
   st.image("./img/Heard1.jpg")

with col2:
   st.header("no history of heart disease")
   st.image("./img/Heard2.png")

with col3:
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

# จัดวางช่องรับข้อมูลเป็น 2 คอลัมน์ ซ้าย-ขวา
in_col1, in_col2 = st.columns(2)

with in_col1:
    age = st.number_input("กรุณากรอกอายุ (Age) [1 - 120 ปี]",min_value=1,max_value=120,value=45,step=1,)

    sex = st.number_input("กรุณากรอกเพศ (Sex) [0 = หญิง, 1 = ชาย]",min_value=0,max_value=1,value=1,step=1,)

with in_col2:
    cp = st.number_input("อาการเจ็บหน้าอก (ChestPainType)",min_value=0,max_value=3,value=1,step=1,)

    hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR) [50 - 250 ครั้ง/นาที]",min_value=50,max_value=250,value=150,step=1,)

# แสดงข้อความสรุปค่าที่กรอกล่าสุดเพื่อตรวจสอบ
st.caption(
    f"ข้อมูลที่เลือกปัจจุบัน -> อายุ: {age} ปี | เพศ: {sex} | อาการเจ็บหน้าอก: {cp} | ชีพจรสูงสุด:"
    f" {hr}"
)


if st.button("ทำนายผล"):
    # 1. แยกข้อมูลสอนโมเดล (X = 4 คอลัมน์, y = คอลัมน์คำตอบ HeartDisease)
    feature_cols = ["Age", "Sex", "ChestPainType", "MaxHR"]
    X = dt[feature_cols]
    y = dt["HeartDisease"]

    # 2. สร้างและฝึกสอนโมเดล K-Nearest Neighbors
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    # 3. นำค่าที่รับมาจากหน้าเว็บมาจัดเรียงเป็น Array 2 มิติ และทำนายผล
    x_input = np.array([[age, sex, cp, hr]])
    out = Knn_model.predict(x_input)
    st.write("ผลการทำนาย (0 = ปกติ, 1 = เสี่ยงโรคหัวใจ):", out[0])

    # 4. ตรวจสอบเงื่อนไขผลลัพธ์ (ใช้ st.image โดยตรง และจับคู่รูปให้ตรงกับ col1/col2)
    if out[0] == 0:
        st.image("./img/Heard2.png")  # 0 = ไม่เป็นโรคหัวใจ
    else:
        st.image("./img/Heard1.jpg")  # 1 = เป็นโรคหัวใจ
else:
    st.write("กรุณากรอกข้อมูลด้านบนให้ครบถ้วน แล้วกดปุ่ม 'ทำนายผล'")