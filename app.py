import streamlit as st

st.set_page_config(page_title="Mon application Streamlit", page_icon="🚀", layout="centered")

st.title("🚀 Mon application Streamlit")
st.write("Bienvenue ! Cette application Python est prête à être personnalisée.")

nom = st.text_input("Comment vous appelez-vous ?")
if st.button("Valider"):
    if nom.strip():
        st.success(f"Bonjour {nom.strip()} !")
    else:
        st.warning("Veuillez saisir votre nom.")
