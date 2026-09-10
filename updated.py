import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SolarGuard",
    page_icon="☀️",
    layout="wide"
)


# =========================================================
# DEVICE
# =========================================================
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "solarguard_resnet18.pth",
        map_location=device
    )

    class_names = checkpoint["class_names"]

    model = resnet18(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, len(class_names))
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, class_names


model, class_names = load_model()


# =========================================================
# IMAGE TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# CONDITION INFORMATION
# =========================================================
condition_info = {

    "Clean": {
        "severity": "Normal",
        "icon": "✅",
        "description":
            "The solar panel appears clean and free from visible surface defects.",
        "impact":
            "The panel is expected to operate normally without major obstruction.",
        "recommendation":
            "No immediate maintenance is required. Continue routine inspection."
    },

    "Dusty": {
        "severity": "Moderate",
        "icon": "🌫️",
        "description":
            "Dust accumulation is detected across the panel surface.",
        "impact":
            "Dust can reduce sunlight absorption and lower power generation efficiency.",
        "recommendation":
            "Schedule panel cleaning using suitable solar-panel cleaning procedures."
    },

    "Bird-drop": {
        "severity": "Moderate",
        "icon": "🐦",
        "description":
            "Bird droppings or localized contamination are detected.",
        "impact":
            "Localized shading may reduce output and may contribute to hot spots.",
        "recommendation":
            "Clean the affected region and inspect the surrounding panel surface."
    },

    "Electrical-damage": {
        "severity": "High",
        "icon": "⚡",
        "description":
            "Visual patterns associated with possible electrical damage are detected.",
        "impact":
            "Electrical defects may reduce output and could indicate internal panel problems.",
        "recommendation":
            "Arrange an immediate inspection by a qualified solar or electrical technician."
    },

    "Physical-Damage": {
        "severity": "High",
        "icon": "🛠️",
        "description":
            "Possible cracks, breakage or structural defects are detected.",
        "impact":
            "Physical damage can reduce efficiency and may lead to further deterioration.",
        "recommendation":
            "Inspect the panel immediately. Repair or replacement may be required."
    },

    "Snow-Covered": {
        "severity": "Moderate",
        "icon": "❄️",
        "description":
            "The panel appears partially or completely covered by snow.",
        "impact":
            "Snow blocks sunlight and can significantly reduce power production.",
        "recommendation":
            "Safely remove the snow using recommended solar-panel maintenance practices."
    }
}


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("☀️ SolarGuard")

st.sidebar.caption(
    "AI-assisted Solar Panel Inspection"
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Defect Detection",
        "Condition Guide",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.write("### System")

st.sidebar.write("Model: **ResNet18**")
st.sidebar.write("Input: **244 × 244**")
st.sidebar.write("Classes: **6**")

st.sidebar.caption(
    "Deep learning classification system for preliminary panel inspection."
)


# =========================================================
# DASHBOARD
# =========================================================
if menu == "Dashboard":

    st.title("☀️ SolarGuard")

    st.subheader(
        "Intelligent Solar Panel Condition Monitoring"
    )

    st.write(
        """
        SolarGuard analyzes solar panel images and identifies visible
        operating conditions that may require cleaning, inspection,
        maintenance, or repair.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # INTRO LAYOUT
    # -----------------------------------------------------
    left, right = st.columns([1.4, 1])

    with left:

        st.subheader("How SolarGuard Works")

        st.write(
            """
            Upload a solar panel image and the trained deep learning model
            evaluates the visual condition of the panel.

            The system provides:

            • Predicted panel condition  
            • Prediction confidence  
            • Condition severity  
            • Potential operational impact  
            • Maintenance recommendation
            """
        )

        if st.button(
            "Go to Defect Detection",
            use_container_width=True
        ):
            st.info(
                "Select 'Defect Detection' from the sidebar to analyze an image."
            )

    with right:

        st.subheader("Inspection Workflow")

        st.info("1️⃣ Upload solar panel image")

        st.info("2️⃣ AI model analyzes image")

        st.info("3️⃣ Condition is classified")

        st.info("4️⃣ Maintenance recommendation is generated")

    st.divider()

    # -----------------------------------------------------
    # CONDITION OVERVIEW
    # -----------------------------------------------------
    st.subheader("Supported Panel Conditions")

    row1 = st.columns(3)

    conditions = list(condition_info.keys())

    for col, condition in zip(
        row1,
        conditions[:3]
    ):

        info = condition_info[condition]

        with col:

            st.markdown(
                f"### {info['icon']} {condition}"
            )

            st.write(
                f"**Severity:** {info['severity']}"
            )

            st.caption(
                info["description"]
            )

    row2 = st.columns(3)

    for col, condition in zip(
        row2,
        conditions[3:]
    ):

        info = condition_info[condition]

        with col:

            st.markdown(
                f"### {info['icon']} {condition}"
            )

            st.write(
                f"**Severity:** {info['severity']}"
            )

            st.caption(
                info["description"]
            )

    st.divider()

    # -----------------------------------------------------
    # MAINTENANCE PRIORITY
    # -----------------------------------------------------
    st.subheader("Maintenance Priority Overview")

    priority_df = pd.DataFrame({
        "Condition": [
            "Clean",
            "Dusty",
            "Bird-drop",
            "Snow-Covered",
            "Electrical-damage",
            "Physical-Damage"
        ],
        "Priority": [
            "Routine",
            "Medium",
            "Medium",
            "Medium",
            "High",
            "High"
        ]
    })

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "SolarGuard helps prioritize panel inspection and maintenance based on visual condition."
    )


# =========================================================
# DEFECT DETECTION
# =========================================================
elif menu == "Defect Detection":

    st.title("🔍 Defect Detection")

    st.write(
        """
        Upload a solar panel image for AI-assisted condition classification.
        """
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a solar panel image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:

        st.info(
            "Upload an image to begin analysis."
        )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        input_tensor = transform(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = class_names[
            prediction.item()
        ]

        confidence_score = (
            confidence.item() * 100
        )

        info = condition_info[
            predicted_class
        ]

        # -------------------------------------------------
        # MAIN ANALYSIS AREA
        # -------------------------------------------------
        image_col, result_col = st.columns(
            [1.1, 1.4]
        )

        with image_col:

            st.subheader("Panel Image")

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                uploaded_file.name
            )

        with result_col:

            st.subheader("AI Inspection Result")

            st.markdown(
                f"## {info['icon']} {predicted_class}"
            )

            result1, result2 = st.columns(2)

            with result1:

                st.metric(
                    "Confidence",
                    f"{confidence_score:.2f}%"
                )

            with result2:

                st.metric(
                    "Severity",
                    info["severity"]
                )

            st.write("**Confidence Level**")

            st.progress(
                min(
                    confidence_score / 100,
                    1.0
                )
            )

            if info["severity"] == "Normal":

                st.success(
                    "Panel condition appears normal."
                )

            elif info["severity"] == "High":

                st.error(
                    "Priority inspection is recommended."
                )

            else:

                st.warning(
                    "Maintenance attention is recommended."
                )

        st.divider()

        # -------------------------------------------------
        # INSPECTION DETAILS
        # -------------------------------------------------
        st.subheader("Inspection Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.markdown("#### Condition Assessment")

            st.write(
                info["description"]
            )

            st.markdown("#### Potential Impact")

            st.write(
                info["impact"]
            )

        with summary_col2:

            st.markdown("#### Recommended Action")

            if info["severity"] == "High":

                st.error(
                    info["recommendation"]
                )

            elif info["severity"] == "Moderate":

                st.warning(
                    info["recommendation"]
                )

            else:

                st.success(
                    info["recommendation"]
                )

            if confidence_score >= 80:

                st.write(
                    "**Prediction Reliability:** High"
                )

            elif confidence_score >= 60:

                st.write(
                    "**Prediction Reliability:** Moderate"
                )

            else:

                st.write(
                    "**Prediction Reliability:** Low"
                )

        st.divider()

        # -------------------------------------------------
        # PROBABILITY ANALYSIS
        # -------------------------------------------------
        st.subheader(
            "AI Confidence Analysis"
        )

        probability_values = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
            * 100
        )

        prob_df = pd.DataFrame({
            "Condition": class_names,
            "Probability": probability_values
        })

        prob_df = prob_df.sort_values(
            "Probability",
            ascending=True
        )

        fig = px.bar(
            prob_df,
            x="Probability",
            y="Condition",
            orientation="h",
            text_auto=".1f"
        )

        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if confidence_score < 60:

            st.warning(
                "Low-confidence prediction. Manual inspection is strongly recommended."
            )


# =========================================================
# CONDITION GUIDE
# =========================================================
elif menu == "Condition Guide":

    st.title("📘 Condition & Maintenance Guide")

    st.write(
        """
        This section provides reference information about the
        six solar panel conditions recognized by SolarGuard.
        """
    )

    st.divider()

    for condition, info in condition_info.items():

        with st.expander(
            f"{info['icon']} {condition} — {info['severity']}"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "**Condition Description**"
                )

                st.write(
                    info["description"]
                )

                st.write(
                    "**Potential Impact**"
                )

                st.write(
                    info["impact"]
                )

            with col2:

                st.write(
                    "**Maintenance Recommendation**"
                )

                st.write(
                    info["recommendation"]
                )


# =========================================================
# ABOUT
# =========================================================
elif menu == "About Project":

    st.title("ℹ️ About SolarGuard")

    st.write(
        """
        SolarGuard is an image-classification system developed for
        automated preliminary inspection of photovoltaic solar panels.

        It uses deep learning and transfer learning to identify visual
        conditions that may affect solar-panel efficiency and maintenance.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Deep Learning Model")

        st.write(
            """
            **Architecture:** ResNet18

            **Approach:** Transfer Learning

            **Fine-tuning:** ResNet layer4

            **Input Resolution:** 244 × 244 pixels

            **Output Classes:** 6
            """
        )

    with col2:

        st.subheader("Recognized Conditions")

        st.write(
            """
            ✅ Clean

            🌫️ Dusty

            🐦 Bird-drop

            ⚡ Electrical-damage

            🛠️ Physical-Damage

            ❄️ Snow-Covered
            """
        )

    st.divider()

    st.subheader("Model Architecture")

    st.code(
        """
Solar Panel Image
        ↓
Resize to 244 × 244
        ↓
ImageNet Normalization
        ↓
Pretrained ResNet18
        ↓
Feature Extraction
        ↓
Dense Layer (512 → 256)
        ↓
ReLU
        ↓
Dropout (0.5)
        ↓
Dense Layer (256 → 6)
        ↓
Solar Panel Condition
        """
    )

    st.divider()

    st.info(
        """
        SolarGuard is designed as an AI-assisted inspection system.
        Predictions involving electrical or physical damage should
        be verified by qualified maintenance personnel.
        """
    )