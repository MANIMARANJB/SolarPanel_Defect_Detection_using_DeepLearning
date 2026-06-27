import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(

    page_title="SolarGuard",

    page_icon="☀️",

    layout="centered"

)


# -----------------------------
# Title
# -----------------------------

st.title(
    "☀️ SolarGuard: Solar Panel Defect Detection"
)


st.write(
    "AI-powered solar panel condition classification using MobileNetV2"
)


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "solarguard_mobilenetv2.keras"
    )

    return model


model = load_model()



# -----------------------------
# Classes
# -----------------------------

class_names = [

    "Bird-drop",

    "Clean",

    "Dusty",

    "Electrical-damage",

    "Physical-Damage",

    "Snow-Covered"

]


# -----------------------------
# Image Upload
# -----------------------------

uploaded_file = st.file_uploader(

    "Upload Solar Panel Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)



if uploaded_file is not None:


    image = Image.open(
        uploaded_file
    )


    st.image(

        image,

        caption="Uploaded Solar Panel Image",

        use_container_width=True

    )


    # -----------------------------
    # Preprocessing
    # -----------------------------

    img = image.resize(
        (224,224)
    )


    img_array = np.array(
        img
    )


    img_array = img_array / 255.0


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(
        img_array
    )


    predicted_class = class_names[
        np.argmax(prediction)
    ]


    confidence = np.max(
        prediction
    ) * 100



    # -----------------------------
    # Display Result
    # -----------------------------

    st.subheader(
        "Prediction Result"
    )


    st.success(
        f"Condition: {predicted_class}"
    )


    st.info(
        f"Confidence: {confidence:.2f}%"
    )



    # -----------------------------
    # Maintenance Recommendation
    # -----------------------------


    recommendations = {


        "Clean":
        "Panel condition is good. Continue regular monitoring.",


        "Dusty":
        "Cleaning recommended to improve solar efficiency.",


        "Bird-drop":
        "Remove bird droppings to avoid reduced energy generation.",


        "Electrical-damage":
        "Immediate inspection required for electrical faults.",


        "Physical-Damage":
        "Inspect panel surface for cracks or structural damage.",


        "Snow-Covered":
        "Remove snow coverage for better sunlight absorption."

    }



    st.write(
        "### Maintenance Recommendation"
    )


    st.warning(
        recommendations[predicted_class]
    )