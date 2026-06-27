import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
import pandas as pd
import plotly.express as px


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SolarGuard AI",
    page_icon="☀️",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size:40px;
        font-weight:bold;
        color:#FFA500;
    }

    .sub-title {
        font-size:20px;
        color:#555;
    }

    .prediction-box {
        padding:20px;
        border-radius:10px;
        background-color:#f5f5f5;
        text-align:center;
        font-size:25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "SolarGuard_MobileNetV2.keras"
    )

    return model


model = load_model()


# =====================================================
# CLASS INFORMATION
# =====================================================


class_names = [

    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"

]


class_description = {


"Clean":
"""
Solar panel is in good condition.

Impact:
Maximum energy generation efficiency.
""",


"Dusty":
"""
Dust accumulation detected on panel surface.

Impact:
Can reduce sunlight absorption and decrease power output.
""",


"Bird-drop":
"""
Bird droppings detected.

Impact:
Blocks sunlight and may create uneven heating.
""",


"Electrical-damage":
"""
Electrical faults such as burn marks or wiring issues detected.

Impact:
Requires immediate inspection to prevent failure.
""",


"Physical-Damage":
"""
Physical damage such as cracks or broken surface detected.

Impact:
May permanently reduce panel performance.
""",


"Snow-Covered":
"""
Snow coverage detected.

Impact:
Blocks sunlight and reduces electricity generation.
"""

}



recommendations = {


"Clean":
"✅ Continue regular monitoring. No immediate maintenance required.",


"Dusty":
"🧹 Schedule cleaning soon to restore panel efficiency.",


"Bird-drop":
"🐦 Remove bird droppings and inspect affected area.",


"Electrical-damage":
"⚡ Immediate technical inspection recommended.",


"Physical-Damage":
"🔧 Inspect panel structure and consider replacement if severe.",


"Snow-Covered":
"❄️ Remove snow coverage to improve sunlight absorption."

}



# =====================================================
# SIDEBAR
# =====================================================

page = st.sidebar.radio(

    "Navigation",

    [
        "🏠 Home",
        "🔍 Defect Prediction",
        "📊 EDA Analysis",
        "💡 Project Insights",
        "ℹ️ About"

    ]

)



# =====================================================
# HOME PAGE
# =====================================================


if page == "🏠 Home":


    st.markdown(
        "<div class='main-title'>☀️ SolarGuard AI</div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='sub-title'>
        Intelligent Solar Panel Defect Detection using Deep Learning
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    col1,col2,col3 = st.columns(3)


    with col1:

        st.metric(
            "Classes Detected",
            "6"
        )


    with col2:

        st.metric(
            "Model",
            "MobileNetV2"
        )


    with col3:

        st.metric(
            "Task",
            "Image Classification"
        )


    st.divider()


    st.subheader("Project Objective")


    st.write(
        """
        SolarGuard automatically identifies solar panel conditions
        using Artificial Intelligence.

        The model classifies images into:

        - Clean
        - Dusty
        - Bird-drop
        - Electrical Damage
        - Physical Damage
        - Snow Covered

        This helps reduce manual inspection time and improves
        solar maintenance decisions.
        """
    )



# =====================================================
# PREDICTION PAGE
# =====================================================


elif page == "🔍 Defect Prediction":


    st.header(
        "🔍 Solar Panel Condition Prediction"
    )


    uploaded_file = st.file_uploader(

        "Upload Solar Panel Image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )


    if uploaded_file:


        image = Image.open(uploaded_file)


        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )


        img = image.resize(
            (224,224)
        )


        img_array = np.array(img)


        img_array = img_array / 255.0


        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        prediction = model.predict(
            img_array
        )


        predicted_class = class_names[
            np.argmax(prediction)
        ]


        confidence = np.max(prediction)*100



        st.divider()


        st.subheader(
            "Prediction Result"
        )


        st.markdown(
            f"""
            <div class='prediction-box'>
            {predicted_class}
            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        st.progress(
            int(confidence)/100
        )


        st.write(
            f"Confidence: {confidence:.2f}%"
        )


        st.subheader(
            "📖 Class Description"
        )


        st.info(
            class_description[predicted_class]
        )


        st.subheader(
            "💡 Maintenance Recommendation"
        )


        st.warning(
            recommendations[predicted_class]
        )



# =====================================================
# EDA PAGE
# =====================================================


elif page == "📊 EDA Analysis":


    st.header(
        "📊 Exploratory Data Analysis"
    )


    class_count = pd.DataFrame(

        {
            "Class":
            class_names,

            "Images":
            [
                150,
                150,
                150,
                150,
                150,
                150
            ]

        }

    )


    fig = px.bar(

        class_count,

        x="Class",

        y="Images",

        title="Class Distribution"

    )


    st.plotly_chart(fig)



    intensity = pd.DataFrame(

        {

        "Class":
        class_names,


        "Average Pixel Intensity":
        [
            120.22,
            109.53,
            124.87,
            130.59,
            114.75,
            114.89
        ]

        }

    )


    fig2 = px.bar(

        intensity,

        x="Class",

        y="Average Pixel Intensity",

        title="Average Pixel Intensity by Class"

    )


    st.plotly_chart(fig2)



# =====================================================
# INSIGHTS PAGE
# =====================================================


elif page == "💡 Project Insights":


    st.header(
        "💡 Business Insights"
    )


    st.success(
        """
        ✔ Automated solar panel inspection

        ✔ Reduces manual monitoring cost

        ✔ Helps schedule cleaning and repair

        ✔ Improves solar energy production

        ✔ Supports smart solar farms
        """
    )


    st.subheader(
        "Key Findings"
    )


    st.write(

        """
        • Electrical damage shows higher pixel intensity variation.

        • Dust accumulation is one of the common efficiency reducing factors.

        • AI-based classification helps prioritize maintenance activities.

        • Early defect detection prevents large repair costs.

        """

    )



# =====================================================
# ABOUT PAGE
# =====================================================


else:


    st.header(
        "ℹ️ About SolarGuard"
    )


    st.write(

        """
        Project:

        SolarGuard: Intelligent Defect Detection on Solar Panels using Deep Learning


        Model:

        MobileNetV2 Transfer Learning


        Framework:

        TensorFlow + Streamlit


        Domain:

        Renewable Energy and Computer Vision

        """

    )