import streamlit as st

st.set_page_config(
    page_title="Pokédex Application",
    page_icon="⚡",
    layout="wide"
)

page_bg_img = '''
<style>
.stApp {
background-image: url("https://images.unsplash.com/photo-1638613067237-b1127ef06c00?auto=format&fit=crop&q=80&w=2982&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
background-repeat: no-repeat;
background-position: center;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('⚡ Welcome to the Pokédex Application')

st.markdown("""
### About This Application
This interactive Pokédex allows you to explore, analyze, and classify Pokémon using various tools and machine learning.

### Available Pages

#### 🔍 **Explore**
Browse and explore detailed Pokémon information including:
- Basic information (type, abilities, height, weight)
- Base stats visualization
- Radar charts

#### 📊 **StatSearch**
Find Pokémon based on statistical criteria:
- Search by base stat ranges (HP, Attack, Defense, etc.)
- Find Pokémon with similar stats
- Compare multiple Pokémon

#### 🤖 **CNN**
Use deep learning for Pokémon image classification:
- Upload Pokémon images
- Get AI-powered predictions
- View confidence scores

### Getting Started
👈 **Select a page from the sidebar to begin your journey!**

---

*Pokémon, electronic game series from Nintendo that debuted in Japan in February 1996 as Pokémon Green and Pokémon Red.*
""")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
