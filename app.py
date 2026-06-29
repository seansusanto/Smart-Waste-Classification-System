import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Smart Waste Classification System",
    page_icon="🗑️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.sidebar.title("🗑️ Smart Waste Classification System")
st.sidebar.markdown("### Model")
st.sidebar.success("YOLO11s Classification")
st.sidebar.markdown("### Classes")
st.sidebar.info("10 Waste Categories")
st.sidebar.markdown("### Developed by Group 9:")
st.sidebar.write("• Joshua Edward Soesito - 2902614905")
st.sidebar.write("• Nathaniel Owen Lim - 2902588365")
st.sidebar.write("• Richardo Imanuel Djari - 2902594475")
st.sidebar.write("• Sean Julian Susanto - 2902594784")
st.sidebar.write("• Wincent - 2902593466")

st.title("🗑️ Smart Waste Classification System")
st.write("Upload an image and let AI identify the waste category.")

descriptions = {
    "plastic": (
        "Plastic waste such as bottles, bags, and food packaging.",
        "Clean before recycling."
    ),

    "glass": (
        "Glass bottles and jars.",
        "Separate by color if possible."
    ),

    "paper": (
        "Paper, newspapers, books.",
        "Keep dry before recycling."
    ),

    "metal": (
        "Metal cans and aluminum containers.",
        "Rinse before disposal."
    ),

    "battery": (
        "Used batteries.",
        "Dispose at hazardous waste collection points."
    ),

    "biological": (
        "Organic waste such as food scraps.",
        "Suitable for composting."
    ),

    "cardboard": (
        "Cardboard boxes.",
        "Flatten before recycling."
    ),

    "clothes": (
        "Used clothes and textiles.",
        "Donate if still usable."
    ),

    "shoes": (
        "Footwear.",
        "Reuse or donate when possible."
    ),

    "trash": (
        "General waste.",
        "Dispose in the appropriate trash bin."
    )
}

image = None

mode = st.radio(
    "**Choose Input Method:**",
    ["Upload Image", "Webcam"]
)

if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif mode == "Webcam":
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image)

if image is not None:

    st.image(image, width='stretch')

    with st.spinner("Analyzing image..."):
        results = model.predict(image, verbose=False)

    if results and results[0].probs is not None:
        probs = results[0].probs
    else:
        st.error("Prediction failed. Please try another image.")
        st.stop()

    top1 = probs.top1
    confidence = probs.top1conf.item()
    prediction = model.names[top1]

    st.divider()
    st.subheader("🔍 Prediction")
    st.success(prediction.upper())

    st.metric("Confidence", f"{confidence*100:.2f}%")

    info = descriptions.get(prediction.lower())

    if info:
        st.divider()
        st.subheader("♻️ Waste Information")
        st.write(info[0])
        st.info(info[1])

    st.divider()
    st.subheader("📊 Prediction Ranking")

    all_probs = probs.data.tolist()

    ranking = [
        (model.names[i], conf)
        for i, conf in enumerate(all_probs)
        if conf > 0.00009
    ]

    ranking.sort(key=lambda x: x[1], reverse=True)

    medals = ["🥇", "🥈", "🥉"]

    for rank, (name, conf) in enumerate(ranking):

        if rank < 3:
            title = medals[rank]
        else:
            title = f"{rank+1}."

        st.write(f"**{title} {name.capitalize()}**")
        st.progress(float(conf))
        st.caption(f"{conf*100:.2f}%")

else:
    st.info("👆 Please upload or take a photo to start prediction.")