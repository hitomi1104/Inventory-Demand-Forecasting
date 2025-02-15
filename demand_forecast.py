import streamlit as st

st.title("Hello, Streamlit in PyCharm! 🎈")
st.write("This is a simple web app running in PyCharm.")

# Adding input elements
name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello, {name}!")

# Adding a button
if st.button("Click me"):
    st.write("Button clicked! 🚀")
