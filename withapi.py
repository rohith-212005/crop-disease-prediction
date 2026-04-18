from flask import Flask, render_template, request 
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini import
from google import genai

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

model = tf.keras.models.load_model("model/diseasemodel.h5")

classes = [
"Potato Healthy",
"Tomato Mosaic Virus"
]

treatments = {
"Potato Healthy":"Plant is healthy.",
"Tomato Mosaic Virus":"Remove infected plants and control whiteflies to prevent spread."
}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict",methods=["POST"])
def predict():

    file = request.files["image"]

    path = os.path.join("static",file.filename)
    file.save(path)

    img = Image.open(path)
    img = img.resize((224,224))

    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    result = classes[np.argmax(prediction)]

    treatment = treatments[result]

    # Gemini AI advice
    prompt = f"""
The detected crop condition is: {result}.

Explain briefly:
1. What this disease means
2. Recommended pesticides
3. Treatment steps
4. Prevention tips for farmers
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    ai_advice = response.text


    return render_template(
        "result.html",
        result=result,
        treatment=treatment,
        image_path=path,
        ai_advice=ai_advice
    )


if __name__ == "__main__":
    app.run(debug=True)