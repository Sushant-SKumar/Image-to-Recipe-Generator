import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Loading the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Recipes List
recipes = {
    'pizza': "🍕 Pizza Recipe:\n- Dough\n- Tomato Sauce\n- Cheese\n- Toppings (Pepperoni, Vegetables)\n- Bake at 220°C for 15-20 minutes.",
    'hotdog': "🌭 Hotdog Recipe:\n- Hotdog Bun\n- Sausage\n- Mustard/Ketchup\n- Onions, Pickles (optional).",
    'ice cream': "🍨 Ice Cream Recipe:\n- Milk\n- Sugar\n- Cream\n- Vanilla Extract\n- Freeze and churn until smooth.",
    'spaghetti': "🍝 Spaghetti Recipe:\n- Spaghetti Pasta\n- Tomato Sauce\n- Garlic\n- Olive Oil\n- Parmesan Cheese.",
    'salad': "🥗 Salad Recipe:\n- Lettuce\n- Tomato\n- Cucumber\n- Olive Oil\n- Lemon Juice.",
    'burger': "🍔 Burger Recipe:\n- Bun\n- Patty (Beef/Veggie)\n- Cheese\n- Lettuce, Tomato\n- Condiments (Ketchup, Mustard).",
    'sushi': "🍣 Sushi Recipe:\n- Sushi Rice\n- Nori Sheets\n- Fish (Salmon, Tuna) or Veggies\n- Soy Sauce, Wasabi.",
    'pancake': "🥞 Pancake Recipe:\n- Flour\n- Milk\n- Eggs\n- Sugar\n- Maple Syrup for topping.",
    'sandwich': "🥪 Sandwich Recipe:\n- Bread\n- Cheese, Lettuce, Tomato\n- Meat (Ham/Turkey) or Veggies\n- Condiments.",
    'omelette': "🍳 Omelette Recipe:\n- Eggs\n- Salt, Pepper\n- Cheese, Vegetables (optional)\n- Cook on medium heat.",
    'fries': "🍟 Fries Recipe:\n- Potatoes\n- Salt\n- Oil (for frying)\n- Ketchup for dipping.",
    'tacos': "🌮 Tacos Recipe:\n- Tortillas\n- Meat (Chicken/Beef) or Veggies\n- Cheese, Salsa\n- Lettuce, Sour Cream.",
    'noodles': "🍜 Noodles Recipe:\n- Noodles\n- Soy Sauce\n- Vegetables\n- Chicken or Tofu (optional).",
    'dumplings': "🥟 Dumplings Recipe:\n- Dough Wrappers\n- Filling (Meat/Veggie)\n- Soy Sauce, Vinegar for dipping.",
    'steak': "🥩 Steak Recipe:\n- Beef Steak\n- Salt, Pepper\n- Butter, Garlic for basting\n- Cook to desired doneness.",
    'cake': "🍰 Cake Recipe:\n- Flour\n- Sugar\n- Eggs\n- Butter\n- Frosting of choice.",
    'curry': "🍛 Curry Recipe:\n- Chicken or Veggies\n- Curry Powder\n- Coconut Milk\n- Rice on the side.",
    'soup': "🥣 Soup Recipe:\n- Vegetables or Chicken\n- Broth\n- Salt, Pepper\n- Herbs for garnish.",
    'popcorn': "🍿 Popcorn Recipe:\n- Corn Kernels\n- Oil\n- Salt or Flavoring of choice.",
    'coffee': "☕ Coffee Recipe:\n- Coffee Powder\n- Hot Water\n- Milk/Sugar (optional)."
}

# Function to predict the food item in the image


def predict_food(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions


# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .title {
        text-align: center;
        font-size: 50px;
        color: #FF4B4B;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #4B4B4B;
        margin-bottom: 30px;
    }
    .recipe-box {
        background-color: #orange;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title and Description
st.markdown("<div class='title'>Food to Recipe Converter 🍴</div>",
            unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload food images and discover delicious recipes!</div>",
            unsafe_allow_html=True)

# File uploader with columns for better layout
uploaded_files = st.file_uploader("📷 Upload Food Images", type=[
                                  'jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    st.markdown("### Uploaded Images:")
    identified_foods = set()

    # Using columns for better layout of images and predictions
    cols = st.columns(2)

    for idx, file in enumerate(uploaded_files):
        image = Image.open(file)

        with cols[idx % 2]:  # Alternate between columns
            st.image(image, caption='Uploaded Image', use_column_width=True)
            with st.spinner('🔍 Analyzing...'):
                predictions = predict_food(image)

            st.markdown("**Predictions:**")
            for i, (imagenet_id, label, score) in enumerate(predictions):
                st.write(
                    f"{i+1}. {label.capitalize()} ({round(score*100, 2)}%)")
                if label in recipes:
                    identified_foods.add(label)

    # Displaying recipes for identified foods
    if identified_foods:
        st.markdown("### Suggested Recipes:")
        for food in identified_foods:
            st.markdown(
                f"<div class='recipe-box'><h4>{food.capitalize()}</h4><p>{recipes[food]}</p></div>", unsafe_allow_html=True)
    else:
        st.info("No known food items identified for recipes.")
