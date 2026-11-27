
# ==========================================
# PROJECT 01: HANDWRITTEN DIGIT RECOGNITION
# Interactive Streamlit Web Application
# ==========================================


# 1. IMPORT REQUIRED LIBRARIES

import streamlit as st

from streamlit_drawable_canvas import st_canvas

from PIL import Image

from src.cnn_predictor import load_model, predict_image


# 2. CONFIGURE THE STREAMLIT PAGE

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)


# 3. APPLICATION TITLE AND DESCRIPTION

st.title("✍️ Handwritten Digit Recognition")

st.write(
    "Draw a digit between 0 and 9 "
    "and let our trained CNN recognise it."
)

st.divider()


# 4. LOAD THE TRAINED CNN MODEL

@st.cache_resource
def get_model():

    return load_model()


try:

    model = get_model()

except Exception as error:

    st.error(f"Could not load the trained CNN: {error}")

    st.stop()


# 5. INITIALISE THE DRAWING CANVAS STATE

if "canvas_key" not in st.session_state:

    st.session_state.canvas_key = 0


# 6. DISPLAY THE DRAWING CANVAS

st.subheader("Draw your digit below")

canvas_result = st_canvas(

    fill_color="#FFFFFF",

    stroke_width=15,

    stroke_color="#000000",

    background_color="#FFFFFF",

    height=280,

    width=280,

    drawing_mode="freedraw",

    update_streamlit=True,
    return_image_data=True,
    key=f"digit_canvas_{st.session_state.canvas_key}"

)


st.caption(
    "Use your mouse or touchscreen to draw one digit."
)


# 7. CREATE PREDICT AND CLEAR BUTTONS

predict_button, clear_button = st.columns(2)


with predict_button:

    predict_clicked = st.button(
        "🔍 Predict Digit",
        use_container_width=True
    )


with clear_button:

    clear_clicked = st.button(
        "🧹 Clear Canvas",
        use_container_width=True
    )


# 8. CLEAR THE DRAWING CANVAS

if clear_clicked:

    st.session_state.canvas_key += 1

    st.session_state.pop("prediction_result", None)

    st.rerun()


# 9. PROCESS THE USER'S DRAWING

if predict_clicked:

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first.")

    else:

        try:

            # Convert canvas image into a PIL image
            drawing = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("RGB")

            # Make prediction using our trained CNN
            result = predict_image(drawing, model)

            # Store the prediction result
            st.session_state.prediction_result = result

        except ValueError as error:

            st.warning(str(error))

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# 10. DISPLAY THE PREDICTION RESULTS

if "prediction_result" in st.session_state:

    result = st.session_state.prediction_result

    st.divider()

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Digit: {result['predicted_digit']}"
    )

    st.write("### Top 3 Predictions")

    for item in result["top_predictions"]:

        digit = item["digit"]

        confidence = item["confidence"]

        st.write(
            f"**Digit {digit}: {confidence:.2f}%**"
        )

        st.progress(
            min(confidence / 100, 1.0)
        )

    st.write("### Processed Image")

    st.image(
        result["processed_image"],
        caption="Image processed before CNN prediction",
        width=180
    )


# 11. APPLICATION FOOTER

st.divider()

st.caption(
    "AI Engineering Portfolio | Project 01 | "
    "Powered by PyTorch and Streamlit | "
    "Built and Designed by @uchecyril"
)
