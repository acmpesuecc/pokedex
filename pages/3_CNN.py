import streamlit as st
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model


st.set_page_config(page_title="CNN Model", layout="wide")

st.title('🤖 CNN Pokémon Image Classification')

@st.cache_resource
def load_cnn_model():
    return load_model('model.h5')

@st.cache_resource
def load_target_classes():
    with open('target_classes.pickle', 'rb') as handle:
        return pickle.load(handle)

st.markdown("""
### Upload a Pokémon Image
Upload an image of a Pokémon and the CNN model will predict which Pokémon it is.
""")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        model = load_cnn_model()
        target_classes = load_target_classes()
        
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes, 1)
        
        # Display original image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(test_img, caption='Uploaded Image', use_column_width=True, channels="BGR")
        
        # Preprocess image
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        test_img = cv2.resize(test_img, (64, 64))
        test_img = test_img[np.newaxis, ...]
        
        # Make prediction
        with st.spinner('Analyzing image...'):
            output = model.predict(test_img)
            output_index = np.argmax(output)
            output_name = target_classes[output_index][1]
            confidence = output[0][output_index] * 100
        
        # Display results
        st.success('Prediction Complete!')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(label="Predicted Pokémon", value=output_name)
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
        
        # Show top 5 predictions
        st.subheader('Top 5 Predictions')
        top_5_indices = np.argsort(output[0])[-5:][::-1]
        
        prediction_data = []
        for idx in top_5_indices:
            pokemon_name = target_classes[idx][1]
            prob = output[0][idx] * 100
            prediction_data.append({'Pokémon': pokemon_name, 'Confidence (%)': f'{prob:.2f}'})
        
        st.table(prediction_data)
        
    except Exception as e:
        st.error(f'Error processing image: {str(e)}')
        st.info('Please make sure model.h5 and target_classes.pickle files are present in the directory.')

else:
    st.info('👆 Please upload an image to get started')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
