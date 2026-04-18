from flask import Flask, render_template, request
# backend , navigate to hyml pages ,  read from/ of  html page

import tensorflow as tf
# loads ml
import numpy as np
# arrays
from PIL import Image
# used to analysize image

import pandas as pd
# to collect data from csv file
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_PATH = "model/diseasemodel.h5"

GROQ_MODEL = "llama-3.3-70b-versatile"


IMAGE_SIZE = (224, 224)

CLASSES = [
    "Potato Healthy",
    "Tomato Mosaic Virus"
]

TREATMENTS = {
    "Potato Healthy"     : "Your plant is healthy. Maintain proper irrigation.",
    "Potato Early Blight": "Use fungicides and remove infected leaves.",
    "Tomato Mosaic Virus": "Remove infected plants and control insects.",
    "Tomato Late Blight" : "Apply copper-based fungicide immediately."
}

DISEASE_MAPPING = {
    "Potato Healthy"     : "Potato___healthy",
    "Potato Early Blight": "Potato___Early_blight",
    "Tomato Mosaic Virus": "Tomato___Tomato_mosaic_virus",
    "Tomato Late Blight" : "Tomato___Late_blight"
}

MOCK_AI_RESPONSES = {

    "Potato Healthy": (
        "** Visual Analysis:** I can visibly confirm a vibrant green hue with no lesions, curling, or spots on the leaf surface. \n\n"
        "**1. Disease Explanation:** The physical characteristics indicate a perfectly healthy crop. \n"
        "**2. Severity:** None. \n"
        "**3. Precautions:** Continue current irrigation. \n"
        "**4. Pesticides:** None required. \n"
        "**5. Prevention:** Ensure proper spacing for airflow."
    ),

    "Potato Early Blight": (
        "** Visual Analysis:** My visual inspection detects distinct, dark concentric target-like rings and brown spots on the lower foliage. \n\n"
        "**1. Disease Explanation:** This indicates Early Blight caused by the fungus *Alternaria solani*. \n"
        "**2. Severity:** Medium/High. \n"
        "**3. Precautions:** Immediately prune infected lower leaves and avoid overhead watering. \n"
        "**4. Pesticides:** Apply a Copper-based organic fungicide or Chlorothalonil. \n"
        "**5. Prevention:** Practice crop rotation and use certified disease-free seeds."
    ),

    "Tomato Mosaic Virus": (
        "** Visual Analysis:** I visually note severe mottling, yellowing, and twisted/distorted leaf veins in the uploaded image. \n\n"
        "**1. Disease Explanation:** This is the highly contagious Tomato Mosaic Virus (ToMV). \n"
        "**2. Severity:** High (Incurable). \n"
        "**3. Precautions:** You must violently uproot and burn the infected plant immediately. DO NOT compost it. Wash your hands and tools rigorously. \n"
        "**4. Pesticides:** Chemicals do not cure viruses; focus on removing pests that carry it. \n"
        "**5. Prevention:** Disinfect pruning tools and wash hands before handling healthy plants."
    ),

    "Tomato Late Blight": (
        "** Visual Analysis:** The image clearly shows large, irregular water-soaked dark lesions with a slight white fungal growth on the underside. \n\n"
        "**1. Disease Explanation:** This is Late Blight, a fast-spreading disease caused by *Phytophthora infestans*. \n"
        "**2. Severity:** Extremely High. \n"
        "**3. Precautions:** Destroy infected plants immediately; keep the foliage dry. \n"
        "**4. Pesticides:** Mancozeb or Copper fungicides applied immediately. \n"
        "**5. Prevention:** Space plants out heavily, avoid night watering, and monitor during humid conditions."
    )

}




client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

model = tf.keras.models.load_model(MODEL_PATH)




@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    # save uploaded image to static folder
    file = request.files["image"]
    path = os.path.join("static", file.filename)
    file.save(path)


    # open, resize, and preprocess image for mobilenetv2
    img_pil = Image.open(path).convert("RGB")
    img_pil = img_pil.resize(IMAGE_SIZE)

    img_array = np.array(img_pil)
    img_batch = np.expand_dims(img_array, axis=0)
    img = preprocess_input(img_batch.astype(np.float32))


    # run cnn prediction
    prediction = model.predict(img)
    confidence = np.max(prediction)
    result = CLASSES[np.argmax(prediction)]


    
    green_fraction = np.sum(green_dominant) / (IMAGE_SIZE[0] * IMAGE_SIZE[1])

    if green_fraction < GREEN_THRESHOLD:

        return render_template(
            "result.html",
            result = "Error: Unrecognized Image (Not a Leaf)",
            confidence = 0.0,
            treatment = "The uploaded image does not appear to be a recognized plant leaf. Please upload a clear photo of a leaf.",
            image_path = path,
            ai_advice = "AI advice is unavailable because a valid leaf was not detected."
        )


    treatment = TREATMENTS.get(result, "No treatment available.")


    #groq prompt for disease analysis
    prompt = f"""
    You are an expert agricultural AI agent.

    Our basic CNN system detected: *{result}* with an initial confidence of {confidence:.2f}.

    Based on the system detection, please provide:
    1. Disease explanation (simple)
    2. Severity level (Low/Medium/High)
    3. Exact Treatment steps / Precautions the farmer should take
    4. Recommended organic or chemical pesticides
    5. Prevention tips for the future
    """

    try:

        # call groq llm for detailed advice
        response  = client.chat.completions.create(
            messages = [{"role": "user", "content": prompt}],
            model  = GROQ_MODEL
        )

        ai_advice = response.choices[0].message.content

    except Exception as e:

        print(f"GROQ DISEASE API ERROR: {e}")

        # go to mock responses if groq is not working
        if "Error" in result:
            ai_advice = "Cannot give AI advice for unrecognized objects."
        else:
            ai_advice = MOCK_AI_RESPONSES.get(result, "Cannot give AI advice.")


    ai_advice = markdown_to_html(ai_advice)


    #collect supplement info from csv
    supplement_name = ""
    supplement_image = ""
    buy_link  = ""

    csv_disease_name = DISEASE_MAPPING.get(result, "")

    if csv_disease_name:

        try:

            df = pd.read_csv("supplement_info.csv")
            sup_info = df[df["disease_name"] == csv_disease_name]

            if not sup_info.empty:

                supplement_name= str(sup_info["supplement name"].values[0])
                supplement_image = str(sup_info["supplement image"].values[0])
                buy_link = str(sup_info["buy link"].values[0])

        except Exception as e:
            print("Error loading supplement info:", e)


    return render_template(
        "result.html",
        result= result,
        confidence= round(confidence * 100, 2),
        treatment = treatment,
        image_path= path,
        ai_advice  = ai_advice,
        supplement_name  = supplement_name  if supplement_name!= "nan" else "",
        supplement_image = supplement_image if supplement_image!= "nan" else "",
        buy_link = buy_link         if buy_link!= "nan" else ""
    )


@app.route("/crop")
def crop():
    return render_template("crop.html")


@app.route("/recommend_crop", methods=["POST"])
def recommend_crop():

    location  = request.form["location"].lower()
    resources = request.form["resources"].lower()
    last_crop = request.form["last_crop"].lower()

    crop_scores = {}


    def add_score(crop, score):
        # accumulate score for each candidate crop
        crop_scores[crop] = crop_scores.get(crop, 0) + score


    # score by soil type
    if "red soil" in resources: add_score("Groundnut", 3)
    if "black soil" in resources: add_score("Cotton",    3)
    if "sandy" in resources: add_score("Millet",    2)


    # score by water availability
    if "irrigation" in resources: add_score("Rice",   3)
    if "rain" in resources: add_score("Millet", 2)


    # score by crop rotation logic
    if   "cotton" in last_crop: add_score("Groundnut", 4)
    elif "rice" in last_crop: add_score("Wheat",     4)
    elif "groundnut" in last_crop: add_score("Maize",   4)


    # score by region
    if "andhra" in location: add_score("Rice",     2); add_score("Groundnut", 2)
    if "telangana" in location: add_score("Cotton",  2)
    if "punjab" in location: add_score("Wheat",    2)


    # pick top scoring crop, fallback to maize
    best_crop = max(crop_scores, key=crop_scores.get) if crop_scores else "Maize"


    # build groq prompt for crop recommendation
    prompt = f"""
    You are an expert agricultural AI agent.
    A farmer from {location} wants to plant a new crop.
    Their resources/soil type: {resources}.
    Their previous crop was: {last_crop}.
    Our system recommends planting: {best_crop}.

    Based on this, please provide:
    1. Why is {best_crop} a good choice for this situation?
    2. Best practices for planting and growing {best_crop} in {location}.
    3. Potential challenges to watch out for.
    4. Expected yield and preparation tips.
    """

    try:

        response  = client.chat.completions.create(
            messages = [{"role": "user", "content": prompt}],
            model    = GROQ_MODEL
        )

        ai_advice = markdown_to_html(response.choices[0].message.content)

    except Exception as e:

        print(f"GROQ RECOMMENDATION API ERROR: {e}")
        ai_advice = f"AI recommendation unavailable. (Error: {str(e)})"


    return render_template(
        "crop_result.html",
        crop      = best_crop,
        location  = location,
        ai_advice = ai_advice
    )


@app.route("/store")
def store():
    return render_template("store.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/login")
def login():
    return render_template("login.html")




if __name__ == "__main__":
    app.run(debug=True)