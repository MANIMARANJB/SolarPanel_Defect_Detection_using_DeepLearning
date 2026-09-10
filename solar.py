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
    page_title="SolarGuard | Solar Panel Defect Detection",
    page_icon="☀️",
    layout="wide"
)


# =========================================================
# MODEL DEVICE
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
# CLASS INFORMATION
# =========================================================
condition_info = {

    "Clean": {
        "severity": "Normal",
        "description":
            "The solar panel appears clean and free from visible surface defects.",
        "impact":
            "The panel is expected to operate normally without significant obstruction.",
        "recommendation":
            "No immediate maintenance is required. Continue routine inspection and periodic cleaning."
    },

    "Dusty": {
        "severity": "Moderate",
        "description":
            "Dust accumulation is detected across the panel surface.",
        "impact":
            "Dust may reduce sunlight absorption and lower overall energy generation efficiency.",
        "recommendation":
            "Schedule cleaning of the panel surface using appropriate solar-panel maintenance procedures."
    },

    "Bird-drop": {
        "severity": "Moderate",
        "description":
            "Bird droppings or similar localized contamination are detected.",
        "impact":
            "Localized shading may reduce panel output and can potentially create hot spots.",
        "recommendation":
            "Clean the affected region carefully and inspect the surrounding panel surface."
    },

    "Electrical-damage": {
        "severity": "High",
        "description":
            "The model has detected visual patterns associated with possible electrical damage.",
        "impact":
            "Electrical defects can reduce power generation and may indicate serious internal panel issues.",
        "recommendation":
            "Arrange an immediate inspection by a qualified solar or electrical technician."
    },

    "Physical-Damage": {
        "severity": "High",
        "description":
            "Visible physical damage such as cracks, breakage, or structural defects may be present.",
        "impact":
            "Physical damage can reduce efficiency and may expose internal components to further deterioration.",
        "recommendation":
            "Inspect the panel immediately. Repair or replacement may be necessary depending on damage severity."
    },

    "Snow-Covered": {
        "severity": "Moderate",
        "description":
            "The solar panel surface appears to be partially or completely covered by snow.",
        "impact":
            "Snow prevents sunlight from reaching photovoltaic cells and may significantly reduce power generation.",
        "recommendation":
            "Remove snow safely using recommended solar-panel cleaning practices."
    }
}


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("☀️ SolarGuard")

st.sidebar.caption(
    "Intelligent Solar Panel Defect Detection System"
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Defect Detection",
        "Model Performance",
        "Condition Guide",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write("Architecture: **ResNet18**")
st.sidebar.write("Learning: **Transfer Learning**")
st.sidebar.write("Image Size: **244 × 244**")
st.sidebar.write("Classes: **6**")
st.sidebar.write("Validation Accuracy: **87.93%**")


# =========================================================
# DASHBOARD
# =========================================================
if menu == "Dashboard":

    st.title("☀️ SolarGuard")
    st.subheader(
        "Intelligent Solar Panel Defect Detection using Deep Learning"
    )

    st.write(
        """
        SolarGuard uses a fine-tuned ResNet18 deep learning model to identify
        common solar panel conditions from images. The system helps support
        inspection and maintenance decisions by classifying panel images into
        six operational conditions.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Validation Accuracy",
            "87.93%",
            "+6.32%"
        )

    with col2:
        st.metric(
            "Weighted F1 Score",
            "87.78%"
        )

    with col3:
        st.metric(
            "Defect Classes",
            "6"
        )

    with col4:
        st.metric(
            "Training Images",
            "869"
        )

    st.divider()

    st.subheader("Supported Panel Conditions")

    condition_df = pd.DataFrame({
        "Condition": [
            "Clean",
            "Dusty",
            "Bird-drop",
            "Electrical-damage",
            "Physical-Damage",
            "Snow-Covered"
        ],
        "Severity": [
            "Normal",
            "Moderate",
            "Moderate",
            "High",
            "High",
            "Moderate"
        ]
    })

    st.dataframe(
        condition_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Use the 'Defect Detection' section from the sidebar "
        "to upload a solar panel image and receive a prediction."
    )


# =========================================================
# DEFECT DETECTION
# =========================================================
elif menu == "Defect Detection":

    st.title("🔍 Solar Panel Defect Detection")

    st.write(
        """
        Upload a solar panel image below. The model will analyze the image,
        identify the predicted condition, calculate the confidence score,
        and provide a maintenance recommendation.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Solar Panel Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.divider()

        left, right = st.columns([1, 1])

        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------
        with left:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"File: {uploaded_file.name}"
            )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------
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
                1
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
        # RESULTS
        # -------------------------------------------------
        with right:

            st.subheader(
                "Analysis Result"
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Predicted Condition",
                    predicted_class
                )

            with metric2:
                st.metric(
                    "Confidence",
                    f"{confidence_score:.2f}%"
                )

            st.write(
                "**Prediction Confidence**"
            )

            st.progress(
                min(
                    confidence_score / 100,
                    1.0
                )
            )

            st.write(
                f"**Severity:** {info['severity']}"
            )

            if info["severity"] == "Normal":

                st.success(
                    "Panel condition appears normal."
                )

            elif info["severity"] == "High":

                st.error(
                    "High-priority maintenance attention recommended."
                )

            else:

                st.warning(
                    "Maintenance attention may be required."
                )

        st.divider()

        # -------------------------------------------------
        # DETAILED ANALYSIS
        # -------------------------------------------------
        st.subheader(
            "Detailed Condition Analysis"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                "### Condition Description"
            )

            st.write(
                info["description"]
            )

        with col2:

            st.markdown(
                "### Potential Impact"
            )

            st.write(
                info["impact"]
            )

        with col3:

            st.markdown(
                "### Maintenance Recommendation"
            )

            st.write(
                info["recommendation"]
            )

        st.divider()

        # -------------------------------------------------
        # PROBABILITY DISTRIBUTION
        # -------------------------------------------------
        st.subheader(
            "Prediction Probability Distribution"
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
            "Probability (%)": probability_values
        })

        prob_df = prob_df.sort_values(
            "Probability (%)",
            ascending=False
        )

        fig = px.bar(
            prob_df,
            x="Probability (%)",
            y="Condition",
            orientation="h",
            text_auto=".2f",
            title="Model Confidence Across All Conditions"
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            prob_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # CONFIDENCE WARNING
        # -------------------------------------------------
        if confidence_score < 60:

            st.warning(
                """
                Low model confidence detected.
                The prediction should be verified through manual inspection.
                """
            )

        elif confidence_score < 80:

            st.info(
                """
                Moderate model confidence.
                Consider additional visual inspection before taking
                maintenance action.
                """
            )

        else:

            st.success(
                """
                High-confidence prediction generated by the model.
                """
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif menu == "Model Performance":

    st.title(
        "📊 Model Performance"
    )

    st.write(
        """
        The final model uses a pretrained ResNet18 architecture with
        transfer learning and fine-tuning of the final ResNet block.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "87.93%"
        )

    with col2:
        st.metric(
            "Precision",
            "88.27%"
        )

    with col3:
        st.metric(
            "Recall",
            "87.93%"
        )

    with col4:
        st.metric(
            "Weighted F1",
            "87.78%"
        )

    st.divider()

    st.subheader(
        "Class-wise Performance"
    )

    performance_df = pd.DataFrame({

        "Condition": [
            "Bird-drop",
            "Clean",
            "Dusty",
            "Electrical-damage",
            "Physical-Damage",
            "Snow-Covered"
        ],

        "Precision": [
            0.9231,
            0.8000,
            0.8571,
            0.8571,
            0.9000,
            1.0000
        ],

        "Recall": [
            0.9474,
            0.9231,
            0.7895,
            0.9000,
            0.6429,
            0.9600
        ],

        "F1 Score": [
            0.9351,
            0.8571,
            0.8219,
            0.8780,
            0.7500,
            0.9796
        ]
    })

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    performance_long = performance_df.melt(
        id_vars="Condition",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        performance_long,
        x="Condition",
        y="Score",
        color="Metric",
        barmode="group",
        title="Class-wise Precision, Recall and F1 Score"
    )

    fig.update_layout(
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Performance Summary"
    )

    st.success(
        """
        Fine-tuning improved the best validation accuracy from
        81.61% to 87.93%.
        """
    )

    st.warning(
        """
        Physical-Damage currently has the lowest recall among the six
        classes. Increasing the number and diversity of physical-damage
        training images could improve future model performance.
        """
    )


# =========================================================
# CONDITION GUIDE
# =========================================================
elif menu == "Condition Guide":

    st.title(
        "📘 Solar Panel Condition Guide"
    )

    st.write(
        "Reference information for all conditions recognized by SolarGuard."
    )

    for condition, info in condition_info.items():

        with st.expander(
            condition
        ):

            st.write(
                f"**Severity:** {info['severity']}"
            )

            st.write(
                "**Description**"
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

            st.write(
                "**Recommended Action**"
            )

            st.write(
                info["recommendation"]
            )


# =========================================================
# ABOUT PROJECT
# =========================================================
elif menu == "About Project":

    st.title(
        "ℹ️ About SolarGuard"
    )

    st.write(
        """
        **SolarGuard** is a computer-vision deep learning system designed
        to classify the visual condition of solar photovoltaic panels.
        """
    )

    st.subheader(
        "Project Objective"
    )

    st.write(
        """
        The objective is to automate preliminary solar-panel inspection
        using image classification and provide useful maintenance guidance.
        """
    )

    st.subheader(
        "Deep Learning Architecture"
    )

    st.write(
        """
        The project uses ResNet18 pretrained on ImageNet.
        The original classification layer was replaced with a custom
        neural-network head containing:
        """
    )

    st.code(
        """
ResNet18 Feature Extractor
          ↓
Fully Connected Layer (512 → 256)
          ↓
ReLU Activation
          ↓
Dropout (0.5)
          ↓
Fully Connected Layer (256 → 6)
          ↓
Predicted Solar Panel Condition
        """
    )

    st.subheader(
        "Training Strategy"
    )

    st.write(
        """
        • Transfer learning with pretrained ResNet18  
        • Image resizing to 244 × 244  
        • Data augmentation  
        • ImageNet normalization  
        • Class-weighted CrossEntropyLoss  
        • Adam optimizer  
        • Fine-tuning of ResNet layer4  
        • Best validation model checkpoint selection
        """
    )

    st.subheader(
        "Supported Classes"
    )

    st.write(
        """
        Clean, Dusty, Bird-drop, Electrical-damage,
        Physical-Damage and Snow-Covered.
        """
    )

    st.subheader(
        "Final Performance"
    )

    st.write(
        """
        Final Validation Accuracy: **87.93%**

        Weighted Precision: **88.27%**

        Weighted Recall: **87.93%**

        Weighted F1 Score: **87.78%**

        Macro F1 Score: **87.03%**
        """
    )

    st.info(
        """
        SolarGuard is intended as an AI-assisted inspection tool.
        High-risk electrical or physical defects should always be
        verified by qualified maintenance personnel.
        """
    )