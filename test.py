import streamlit as st

st.title("hello，Streamlit 👋")
name = st.text_input("please input your name")
if name:
    st.write(f"welcome,{name}！")
    