import os
import math
import random 
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from dotenv import load_dotenv
import cv2
import google.generativeai as genai
import json
import numpy as np
import base64
from io import BytesIO
# app.py
from food_data import food_categories, nutrition_data, get_foods_by_category, get_nutrition_data

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24).hex()
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['DATABASE'] = 'nutrition_app.db'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Gemini
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-2.0-flash')  # Updated model name
except Exception as e:
    print(f"Error initializing Gemini: {str(e)}")
    model = None

@app.template_filter('format_time')
def format_time(timestamp):
    return datetime.fromisoformat(timestamp).strftime('%H:%M')

def get_nutrition_info(food_item, quantity=100):
    try:
        prompt = f"""Provide accurate nutrition information per 100g for {food_item} in this exact JSON format:
        {{
            "food": "{food_item}",
            "calories_per_100g": number,
            "protein_per_100g": number,
            "carbs_per_100g": number,
            "fats_per_100g": number
        }}
        Return only the JSON with no additional text."""
        
        response = model.generate_content(prompt)
        
        # Clean the response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        data = json.loads(response_text)
        
        # Calculate for user's quantity
        ratio = quantity / 100
        return {
            "food": food_item,
            "quantity": quantity,
            "calories": data['calories_per_100g'] * ratio,
            "protein": data['protein_per_100g'] * ratio,
            "carbs": data['carbs_per_100g'] * ratio,
            "fats": data['fats_per_100g'] * ratio,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting nutrition info: {str(e)}")
        return None

def parse_gemini_response(response_text):
    try:
        # Try to find JSON in the response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        json_str = response_text[start:end]
        return json.loads(json_str)
    except Exception as e:
        app.logger.error(f"Error parsing Gemini response: {str(e)}")
        return {"error": "Could not parse AI response. Please try again."}

def calculate_bmi_status(weight, height):
    """Calculate BMI and return status"""
    height_m = height / 100  # convert cm to m
    bmi = weight / (height_m ** 2)
    
    if bmi < 18.5:
        return "underweight", bmi
    elif 18.5 <= bmi < 25:
        return "normal weight", bmi
    elif 25 <= bmi < 30:
        return "overweight", bmi
    else:
        return "obese", bmi

def calculate_nutritional_targets(user_data):
    """Calculate basic nutritional targets based on user data"""
    try:
        age = int(user_data['age'])
        weight = float(user_data['weight'])
        height = float(user_data['height'])
        activity_level = user_data['activity_level']
        goal = user_data['goal']
        
        # Calculate BMI status
        bmi_status, bmi_value = calculate_bmi_status(weight, height)
        
        # Basic Harris-Benedict equation for BMR
        if user_data['gender'].lower() == 'male':
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        
        # Activity multipliers
        activity_factors = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'extreme': 1.9
        }
        
        tdee = bmr * activity_factors.get(activity_level, 1.2)
        
        # Adjust for goals
        if goal == 'weight_loss':
            tdee *= 0.85
        elif goal == 'weight_gain':
            tdee *= 1.15
        elif goal == 'muscle_gain':
            tdee *= 1.1
        
        # Macronutrient distribution (simplified)
        protein = round(weight * 2.2)  # 2.2g per kg of body weight
        fats = round((tdee * 0.25) / 9)  # 25% of calories from fat
        carbs = round((tdee - (protein * 4) - (fats * 9)) / 4)
        
        return {
            'calories': round(tdee),
            'protein': protein,
            'carbs': carbs,
            'fats': fats,
            'bmi_status': bmi_status,
            'bmi_value': round(bmi_value, 1)
        }
    except Exception as e:
        app.logger.error(f"Error calculating nutritional targets: {str(e)}")
        return {
            'calories': 2000,
            'protein': 100,
            'carbs': 250,
            'fats': 65,
            'bmi_status': 'unknown',
            'bmi_value': 0
        }

# Nutrition data for 80 Indian food items
data_cat_nutrition = {
    'adhirasam': {'calories': 385, 'carbs': 75, 'protein': 4, 'fats': 12, 'vitamins': 2},
    'aloo_gobi': {'calories': 120, 'carbs': 18, 'protein': 4, 'fats': 5, 'vitamins': 6},
    'aloo_matar': {'calories': 160, 'carbs': 25, 'protein': 6, 'fats': 6, 'vitamins': 5},
    'aloo_methi': {'calories': 130, 'carbs': 20, 'protein': 5, 'fats': 5, 'vitamins': 8},
    'aloo_shimla_mirch': {'calories': 140, 'carbs': 22, 'protein': 3, 'fats': 6, 'vitamins': 5},
    'aloo_tikki': {'calories': 200, 'carbs': 30, 'protein': 6, 'fats': 8, 'vitamins': 4},
    'anarsa': {'calories': 350, 'carbs': 65, 'protein': 5, 'fats': 10, 'vitamins': 2},
    'ariselu': {'calories': 380, 'carbs': 70, 'protein': 4, 'fats': 12, 'vitamins': 2},
    'bandar_laddu': {'calories': 450, 'carbs': 75, 'protein': 8, 'fats': 15, 'vitamins': 3},
    'basundi': {'calories': 280, 'carbs': 35, 'protein': 8, 'fats': 12, 'vitamins': 5},
    'bhatura': {'calories': 300, 'carbs': 50, 'protein': 7, 'fats': 10, 'vitamins': 2},
    'bhindi_masala': {'calories': 110, 'carbs': 15, 'protein': 4, 'fats': 5, 'vitamins': 7},
    'biryani': {'calories': 350, 'carbs': 45, 'protein': 15, 'fats': 12, 'vitamins': 4},
    'boondi': {'calories': 400, 'carbs': 70, 'protein': 5, 'fats': 15, 'vitamins': 2},
    'butter_chicken': {'calories': 450, 'carbs': 10, 'protein': 25, 'fats': 30, 'vitamins': 5},
    'chak_hao_kheer': {'calories': 280, 'carbs': 40, 'protein': 7, 'fats': 10, 'vitamins': 4},
    'cham_cham': {'calories': 320, 'carbs': 60, 'protein': 8, 'fats': 8, 'vitamins': 3},
    'chana_masala': {'calories': 180, 'carbs': 25, 'protein': 9, 'fats': 6, 'vitamins': 7},
    'chapati': {'calories': 120, 'carbs': 25, 'protein': 4, 'fats': 2, 'vitamins': 2},
    'chhena_kheeri': {'calories': 250, 'carbs': 30, 'protein': 10, 'fats': 10, 'vitamins': 4},
    'chicken_razala': {'calories': 320, 'carbs': 8, 'protein': 28, 'fats': 18, 'vitamins': 6},
    'chicken_tikka': {'calories': 250, 'carbs': 5, 'protein': 30, 'fats': 12, 'vitamins': 7},
    'chicken_tikka_masala': {'calories': 380, 'carbs': 15, 'protein': 25, 'fats': 25, 'vitamins': 5},
    'chikki': {'calories': 350, 'carbs': 50, 'protein': 7, 'fats': 15, 'vitamins': 3},
    'daal_baati_churma': {'calories': 600, 'carbs': 80, 'protein': 15, 'fats': 25, 'vitamins': 5},
    'daal_puri': {'calories': 280, 'carbs': 35, 'protein': 10, 'fats': 10, 'vitamins': 4},
    'dal_makhani': {'calories': 350, 'carbs': 30, 'protein': 12, 'fats': 20, 'vitamins': 6},
    'dal_tadka': {'calories': 200, 'carbs': 25, 'protein': 10, 'fats': 8, 'vitamins': 7},
    'dharwad_pedha': {'calories': 380, 'carbs': 65, 'protein': 6, 'fats': 12, 'vitamins': 2},
    'doodhpak': {'calories': 300, 'carbs': 45, 'protein': 8, 'fats': 12, 'vitamins': 4},
    'double_ka_meetha': {'calories': 400, 'carbs': 60, 'protein': 6, 'fats': 15, 'vitamins': 3},
    'dum_aloo': {'calories': 220, 'carbs': 25, 'protein': 4, 'fats': 12, 'vitamins': 5},
    'gajar_ka_halwa': {'calories': 350, 'carbs': 50, 'protein': 5, 'fats': 15, 'vitamins': 8},
    'gavvalu': {'calories': 400, 'carbs': 70, 'protein': 5, 'fats': 15, 'vitamins': 2},
    'ghevar': {'calories': 450, 'carbs': 75, 'protein': 6, 'fats': 20, 'vitamins': 2},
    'gulab_jamun': {'calories': 380, 'carbs': 65, 'protein': 5, 'fats': 15, 'vitamins': 2},
    'imarti': {'calories': 350, 'carbs': 70, 'protein': 5, 'fats': 10, 'vitamins': 2},
    'jalebi': {'calories': 400, 'carbs': 80, 'protein': 3, 'fats': 12, 'vitamins': 1},
    'kachori': {'calories': 300, 'carbs': 40, 'protein': 7, 'fats': 15, 'vitamins': 4},
    'kadai_paneer': {'calories': 350, 'carbs': 12, 'protein': 18, 'fats': 25, 'vitamins': 7},
    'kadhi_pakoda': {'calories': 280, 'carbs': 20, 'protein': 10, 'fats': 15, 'vitamins': 5},
    'kajjikaya': {'calories': 380, 'carbs': 60, 'protein': 5, 'fats': 18, 'vitamins': 2},
    'kakinada_khaja': {'calories': 420, 'carbs': 70, 'protein': 4, 'fats': 18, 'vitamins': 2},
    'kalakand': {'calories': 350, 'carbs': 40, 'protein': 8, 'fats': 20, 'vitamins': 4},
    'karela_bharta': {'calories': 100, 'carbs': 15, 'protein': 3, 'fats': 4, 'vitamins': 9},
    'kofta': {'calories': 300, 'carbs': 20, 'protein': 15, 'fats': 18, 'vitamins': 5},
    'kuzhi_paniyaram': {'calories': 200, 'carbs': 30, 'protein': 5, 'fats': 8, 'vitamins': 4},
    'lassi': {'calories': 180, 'carbs': 20, 'protein': 6, 'fats': 8, 'vitamins': 5},
    'ledikeni': {'calories': 380, 'carbs': 60, 'protein': 6, 'fats': 15, 'vitamins': 3},
    'litti_chokha': {'calories': 350, 'carbs': 50, 'protein': 10, 'fats': 15, 'vitamins': 6},
    'lyangcha': {'calories': 400, 'carbs': 70, 'protein': 5, 'fats': 15, 'vitamins': 2},
    'maach_jhol': {'calories': 250, 'carbs': 10, 'protein': 22, 'fats': 15, 'vitamins': 6},
    'makki_di_roti_sarson_da_saag': {'calories': 300, 'carbs': 40, 'protein': 10, 'fats': 12, 'vitamins': 8},
    'malapua': {'calories': 380, 'carbs': 65, 'protein': 5, 'fats': 15, 'vitamins': 3},
    'misi_roti': {'calories': 200, 'carbs': 35, 'protein': 7, 'fats': 6, 'vitamins': 4},
    'misti_doi': {'calories': 150, 'carbs': 20, 'protein': 6, 'fats': 5, 'vitamins': 4},
    'modak': {'calories': 250, 'carbs': 40, 'protein': 5, 'fats': 8, 'vitamins': 3},
    'mysore_pak': {'calories': 450, 'carbs': 60, 'protein': 5, 'fats': 25, 'vitamins': 2},
    'naan': {'calories': 290, 'carbs': 50, 'protein': 9, 'fats': 10, 'vitamins': 2},
    'navrattan_korma': {'calories': 350, 'carbs': 30, 'protein': 12, 'fats': 20, 'vitamins': 7},
    'palak_paneer': {'calories': 220, 'carbs': 15, 'protein': 14, 'fats': 12, 'vitamins': 9},
    'paneer_butter_masala': {'calories': 330, 'carbs': 18, 'protein': 16, 'fats': 22, 'vitamins': 6},
    'phirni': {'calories': 280, 'carbs': 35, 'protein': 7, 'fats': 10, 'vitamins': 4},
    'pithe': {'calories': 320, 'carbs': 55, 'protein': 6, 'fats': 12, 'vitamins': 3},
    'poha': {'calories': 250, 'carbs': 40, 'protein': 6, 'fats': 8, 'vitamins': 5},
    'poornalu': {'calories': 350, 'carbs': 60, 'protein': 5, 'fats': 15, 'vitamins': 3},
    'pootharekulu': {'calories': 400, 'carbs': 70, 'protein': 4, 'fats': 15, 'vitamins': 2},
    'qubani_ka_meetha': {'calories': 280, 'carbs': 50, 'protein': 4, 'fats': 10, 'vitamins': 6},
    'rabri': {'calories': 380, 'carbs': 45, 'protein': 10, 'fats': 20, 'vitamins': 4},
    'ras_malai': {'calories': 320, 'carbs': 40, 'protein': 8, 'fats': 15, 'vitamins': 5},
    'rasgulla': {'calories': 300, 'carbs': 60, 'protein': 6, 'fats': 8, 'vitamins': 3},
    'sandesh': {'calories': 250, 'carbs': 30, 'protein': 8, 'fats': 10, 'vitamins': 4},
    'shankarpali': {'calories': 420, 'carbs': 70, 'protein': 5, 'fats': 20, 'vitamins': 2},
    'sheer_korma': {'calories': 380, 'carbs': 50, 'protein': 8, 'fats': 18, 'vitamins': 4},
    'sheera': {'calories': 350, 'carbs': 60, 'protein': 5, 'fats': 15, 'vitamins': 3},
    'shrikhand': {'calories': 330, 'carbs': 45, 'protein': 7, 'fats': 15, 'vitamins': 4},
    'sohan_halwa': {'calories': 500, 'carbs': 80, 'protein': 6, 'fats': 25, 'vitamins': 2},
    'sohan_papdi': {'calories': 450, 'carbs': 70, 'protein': 5, 'fats': 20, 'vitamins': 2},
    'sutar_feni': {'calories': 520, 'carbs': 90, 'protein': 5, 'fats': 25, 'vitamins': 1},
    'unni_appam': {'calories': 400, 'carbs': 70, 'protein': 4, 'fats': 15, 'vitamins': 3}
}

# Database setup
def get_db_connection():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with open('schema.sql') as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

def setup_database():
    if not os.path.exists(app.config['DATABASE']):
        with app.app_context():
            init_db()

# Create the schema.sql file if it doesn't exist
if not os.path.exists('schema.sql'):
    with open('schema.sql', 'w') as f:
        f.write('''
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    height REAL NOT NULL,
    weight REAL NOT NULL,
    activity TEXT NOT NULL,
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''')

class NutritionAnalyzer:
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    @staticmethod
    def calculate_calories(age, gender, height_cm, weight_kg, activity):
        """Calculate daily calorie needs using Mifflin-St Jeor Equation"""
        if gender.lower() == 'male':
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:  # female
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        
        activity_multipliers = {
            'sedentary': 1.2,
            'lightly': 1.375,
            'moderately': 1.55,
            'very': 1.725,
            'extra': 1.9
        }
        return bmr * activity_multipliers.get(activity.lower(), 1.2)

    @staticmethod
    def get_bmi_category(bmi):
        if bmi < 18.5:
            return "Underweight", "bg-primary"
        elif 18.5 <= bmi < 25:
            return "Normal", "bg-success"
        elif 25 <= bmi < 30:
            return "Overweight", "bg-warning"
        else:
            return "Obese", "bg-danger"

    @staticmethod
    def load_model():
        try:
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(data_cat_nutrition)))
            
            checkpoint = torch.load('resnet50_finetuned_model.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint.get('state_dict', checkpoint))
            model.eval()
            return model
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            return None

    @staticmethod
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path)
        return transform(image).unsqueeze(0)

    @staticmethod
    def predict(image_path, model):
        try:
            img_tensor = NutritionAnalyzer.preprocess_image(image_path)
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_cat = torch.max(probabilities, 1)
                predicted_food = list(data_cat_nutrition.keys())[top_cat.item()]
                return {
                    'food': predicted_food,
                    'confidence': top_prob.item(),
                    'nutrition': data_cat_nutrition[predicted_food],
                    'image': image_path
                }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

# Routes
@app.route('/')
def index():
    return redirect(url_for('landing'))

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/home')
def home():
    if 'user' in session:
        return redirect(url_for('profile'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('profile'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            # Calculate user metrics
            activity = user['activity'].split()[0].lower() if user['activity'] else 'sedentary'
            tdee = NutritionAnalyzer.calculate_calories(user['age'], user['gender'], user['height'], user['weight'], activity)
            bmi = user['weight'] / ((user['height']/100) ** 2)
            bmi_category, bmi_class = NutritionAnalyzer.get_bmi_category(bmi)
            
            # Store user data in session
            session['user'] = {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
                'age': user['age'],
                'gender': user['gender'],
                'height': user['height'],
                'weight': user['weight'],
                'activity': user['activity'],
                'tdee': tdee,
                'bmi': bmi,
                'bmi_category': bmi_category,
                'bmi_class': bmi_class
            }
            
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html', login_page=True)

@app.route('/face_login', methods=['GET', 'POST'])
def face_login():
    if 'user' in session:
        return redirect(url_for('profile'))
        
    if request.method == 'POST':
        try:
            # Get the image from the POST request
            image_data = request.form.get('image').split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                return jsonify({'success': False, 'message': 'No face detected'})
                
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            if True in matches:
                matched_idx = matches.index(True)
                user_id = known_face_ids[matched_idx]
                
                # Get user from database
                conn = get_db_connection()
                user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
                conn.close()
                
                if user:
                    # Calculate user metrics (same as regular login)
                    activity = user['activity'].split()[0].lower() if user['activity'] else 'sedentary'
                    tdee = NutritionAnalyzer.calculate_calories(user['age'], user['gender'], user['height'], user['weight'], activity)
                    bmi = user['weight'] / ((user['height']/100) ** 2)
                    bmi_category, bmi_class = NutritionAnalyzer.get_bmi_category(bmi)
                    
                    # Store user data in session
                    session['user'] = {
                        'id': user['id'],
                        'name': user['name'],
                        'email': user['email'],
                        'age': user['age'],
                        'gender': user['gender'],
                        'height': user['height'],
                        'weight': user['weight'],
                        'activity': user['activity'],
                        'tdee': tdee,
                        'bmi': bmi,
                        'bmi_category': bmi_category,
                        'bmi_class': bmi_class
                    }
                    
                    return jsonify({'success': True, 'redirect': url_for('profile')})
            
            return jsonify({'success': False, 'message': 'Face not recognized'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    return render_template('face_login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('profile'))
        
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            age = int(request.form.get('age'))
            gender = request.form.get('gender')
            height = float(request.form.get('height'))
            weight = float(request.form.get('weight'))
            activity = request.form.get('activity')
            face_data = request.form.get('face_data')  # Get face data if available
            
            # Validate required fields
            if not all([name, email, password, age, gender, height, weight, activity]):
                flash('Please fill all required fields', 'danger')
                return redirect(url_for('register'))
            
            # Check if email exists
            conn = get_db_connection()
            existing_user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if existing_user:
                flash('Email already registered', 'danger')
                conn.close()
                return redirect(url_for('register'))
            
            # Create new user
            password_hash = generate_password_hash(password)
            conn.execute(
                'INSERT INTO users (name, email, password, age, gender, height, weight, activity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (name, email, password_hash, age, gender, height, weight, activity)
            )
            
            # Get the new user's ID
            user_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
            
            # If face data was provided, register it
            if face_data:
                try:
                    image_data = face_data.split(",")[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Find face encodings
                    face_locations = face_recognition.face_locations(image)
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_ids.append(user_id)
                except Exception as e:
                    print(f"Error registering face: {str(e)}")
            
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error processing registration: {str(e)}', 'danger')
    
    return render_template('register.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            user_id = session['user']['id']
            age = int(request.form.get('age'))
            gender = request.form.get('gender')
            height = float(request.form.get('height'))
            weight = float(request.form.get('weight'))
            activity = request.form.get('activity')
            
            # Update database
            conn = get_db_connection()
            conn.execute(
                'UPDATE users SET age = ?, gender = ?, height = ?, weight = ?, activity = ? WHERE id = ?',
                (age, gender, height, weight, activity, user_id)
            )
            conn.commit()
            conn.close()
            
            # Update session data
            activity_level = activity.split()[0].lower() if activity else 'sedentary'
            tdee = NutritionAnalyzer.calculate_calories(age, gender, height, weight, activity_level)
            bmi = weight / ((height/100) ** 2)
            bmi_category, bmi_class = NutritionAnalyzer.get_bmi_category(bmi)
            
            session['user'].update({
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'activity': activity,
                'tdee': tdee,
                'bmi': bmi,
                'bmi_category': bmi_category,
                'bmi_class': bmi_class
            })
            
            flash('Profile updated successfully!', 'success')
        except Exception as e:
            flash(f'Error updating profile: {str(e)}', 'danger')
    
    return render_template('profile.html', user=session['user'])

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('landing'))

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    model = NutritionAnalyzer.load_model()
    if not model:
        flash('Failed to load the model. Please try again later.', 'danger')
        return redirect(url_for('prediction'))
    
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if not file:
            flash('No file uploaded', 'danger')
            return redirect(url_for('prediction'))
        
        if file and NutritionAnalyzer.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Debug: Print upload details
            print(f"\nDEBUG UPLOAD DETAILS:")
            print(f"Original filename: {file.filename}")
            print(f"Secure filename: {filename}")
            print(f"Full filepath: {filepath}")
            print(f"File content type: {file.content_type}")
            print(f"File size: {len(file.read())} bytes")
            file.seek(0)  # Reset file pointer after reading
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save file and verify
            try:
                file.save(filepath)
                print("DEBUG: File saved successfully")
                
                # Verify file exists
                if os.path.exists(filepath):
                    print("DEBUG: File verification passed - exists on disk")
                    print(f"DEBUG: File size on disk: {os.path.getsize(filepath)} bytes")
                else:
                    print("DEBUG ERROR: File save failed - not found on disk")
                    flash('Failed to save image file', 'danger')
                    return redirect(url_for('prediction'))
                
                result = NutritionAnalyzer.predict(filepath, model)
                if result:
                    print("DEBUG PREDICTION RESULT:")
                    print(f"Food: {result['food']}")
                    print(f"Confidence: {result['confidence']}")
                    print(f"Image path in result: {result['image']}")
                    
                    # Store relative path for the template
                    result['image'] = os.path.join('uploads', filename).replace('\\', '/')
                    session['result'] = result
                    print(f"Session result: {session['result']}")
                    flash('Analysis complete!', 'success')
                    return redirect(url_for('nutrition'))
                else:
                    print("DEBUG ERROR: Prediction failed")
                    flash('Failed to analyze image', 'danger')
            except Exception as e:
                print(f"DEBUG ERROR: Exception during file save: {str(e)}")
                flash(f'Error processing image: {str(e)}', 'danger')
        else:
            flash('Invalid file format. Allowed formats: png, jpg, jpeg', 'danger')
    
    return render_template('prediction.html')

@app.route('/nutrition')
def nutrition():
    if 'user' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
        
    if 'result' not in session:
        flash('Please analyze a food image first', 'warning')
        return redirect(url_for('prediction'))
    
    print(f"Rendering nutrition page with session result: {session['result']}")
    return render_template('nutrition.html', 
                         food=session['result']['food'],
                         confidence=session['result']['confidence'],
                         nutrition=session['result']['nutrition'],
                         image=session['result']['image'])

@app.route('/calculator')
def calculator():
    if 'user' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
        
    if 'result' not in session:
        flash('Please analyze a food image first', 'warning')
        return redirect(url_for('prediction'))
    
    user = session['user']
    nutrition = session['result']['nutrition']
    
    daily_percent = (nutrition['calories'] / user['tdee']) * 100
    remaining = user['tdee'] - nutrition['calories']
    protein_needs = user['weight'] * 1.6
    
    return render_template('calculator.html',
                         user=user,
                         nutrition=nutrition,
                         daily_percent=daily_percent,
                         remaining=remaining,
                         protein_needs=protein_needs)

@app.route('/daily_calories')
def daily_calories():
    if 'user' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    user = session['user']
    
    # Calculate all required nutritional values
    weight = user['weight']
    tdee = user['tdee']
    
    # Macronutrient calculations
    protein_grams = weight * 1.6  # 1.6g per kg of body weight
    fat_grams = (tdee * 0.25) / 9  # 25% of calories from fat
    carb_grams = (tdee - (protein_grams * 4 + fat_grams * 9)) / 4  # Remaining calories from carbs
    water_liters = weight * 0.033  # 0.033 * weight in kg
    
    # Different calorie targets
    maintenance = tdee
    weight_loss = tdee - 500  # 500 calorie deficit for 0.5kg/week loss
    weight_gain = tdee + 500  # 500 calorie surplus for muscle gain
    
    return render_template('daily_calories.html',
                         user=user,
                         protein_grams=protein_grams,
                         fat_grams=fat_grams,
                         carb_grams=carb_grams,
                         water_liters=water_liters,
                         maintenance=maintenance,
                         weight_loss=weight_loss,
                         weight_gain=weight_gain)

@app.route('/dietary_plan')
def dietary_plan():
    if 'user' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    user = session['user']
    goal = request.args.get('goal', 'maintenance')
    
    # Get target calories based on goal
    if goal == 'weight_loss':
        target_calories = user['tdee'] - 500
        plan_title = "Weight Loss Meal Plan"
    elif goal == 'weight_gain':
        target_calories = user['tdee'] + 500
        plan_title = "Weight Gain Meal Plan"
    else:
        target_calories = user['tdee']
        plan_title = "Maintenance Meal Plan"

    # Calculate meal calorie distribution
    meal_calories = {
        'breakfast': target_calories * 0.20,
        'lunch': target_calories * 0.30,
        'dinner': target_calories * 0.25,
        'snack1': target_calories * 0.10,
        'snack2': target_calories * 0.10,
        'dessert': target_calories * 0.05
    }
    
    # Get all food options
    breakfast_data = get_foods_by_category('breakfast')
    lunch_data = get_foods_by_category('lunch')
    snack_data = get_foods_by_category('snacks')
    dessert_data = get_foods_by_category('desserts')
    
    breakfast_options = list(breakfast_data.keys())
    main_meal_options = list(lunch_data.keys())
    snack_options = list(snack_data.keys())
    dessert_options = list(dessert_data.keys())
    
    # Combine all nutrition data
    all_nutrition_data = {}
    all_nutrition_data.update(breakfast_data)
    all_nutrition_data.update(lunch_data)
    all_nutrition_data.update(snack_data)
    all_nutrition_data.update(dessert_data)
    
    # Generate daily plan based on current date (changes daily)
    today = datetime.now().date()
    date_seed = int(today.strftime('%Y%m%d'))  # Creates a unique number for each day
    
    # Use a unique identifier from the user - fall back to email if username doesn't exist
    user_identifier = user.get('username') or user.get('email') or 'default_user'
    random.seed(date_seed + hash(user_identifier))  # Make unique per user
    
    # Select today's meals
    today_breakfast = random.choice(breakfast_options)
    today_lunch = random.choice(main_meal_options)
    today_snack1 = random.choice(snack_options)
    today_dinner = random.choice(main_meal_options)
    today_snack2 = random.choice(snack_options)
    today_dessert = random.choice(dessert_options)
    
    # Reset random seed for weekly plan
    random.seed()
    
    # Generate weekly plan with variety
    weekly_plan = []
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for i, day in enumerate(days):
        # Use day index + date seed to create variety but maintain consistency
        day_seed = date_seed + i
        random.seed(day_seed)
        
        daily_plan = {
            'day': day,
            'breakfast': random.choice(breakfast_options),
            'lunch': random.choice(main_meal_options),
            'snack1': random.choice(snack_options),
            'dinner': random.choice(main_meal_options),
            'snack2': random.choice(snack_options),
            'dessert': random.choice(dessert_options)
        }
        weekly_plan.append(daily_plan)
    
    # Reset random seed
    random.seed()
    
    return render_template('dietary_plan.html',
                         user=user,
                         weekly_plan=weekly_plan,
                         nutrition_data=all_nutrition_data,
                         goal=goal,
                         meal_calories=meal_calories,
                         plan_title=plan_title,
                         target_calories=target_calories,
                         breakfast_options=breakfast_options,
                         main_meal_options=main_meal_options,
                         snack_options=snack_options,
                         dessert_options=dessert_options,
                         today_breakfast=today_breakfast,
                         today_lunch=today_lunch,
                         today_snack1=today_snack1,
                         today_dinner=today_dinner,
                         today_snack2=today_snack2,
                         today_dessert=today_dessert)

@app.route('/tracking', methods=['GET', 'POST'])
def tracking():
    if 'user' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))

    if 'food_log' not in session:
        session['food_log'] = []
    
    if request.method == 'POST':
        food_item = request.form.get('food_item', '').strip()
        quantity = float(request.form.get('quantity', 100))
        
        if not food_item:
            flash('Please enter a food item', 'danger')
            return redirect(url_for('tracking'))
        
        # Check if manual entry form was submitted
        if 'manual_submit' in request.form:
            try:
                nutrition_info = {
                    "food": food_item,
                    "quantity": quantity,
                    "calories": float(request.form.get('calories', 0)),
                    "protein": float(request.form.get('protein', 0)),
                    "carbs": float(request.form.get('carbs', 0)),
                    "fats": float(request.form.get('fats', 0)),
                    "timestamp": datetime.now().isoformat()
                }
                session['food_log'].append(nutrition_info)
                session.modified = True
                flash('Manual entry added successfully!', 'success')
                return redirect(url_for('tracking'))
            except ValueError:
                flash('Invalid nutrition values', 'danger')
                return redirect(url_for('tracking'))
        
        # Try to get info from Gemini
        nutrition_info = get_nutrition_info(food_item, quantity)
        
        if nutrition_info:
            session['food_log'].append(nutrition_info)
            session.modified = True
            flash('Food item added successfully!', 'success')
        else:
            # Show manual entry form if API fails
            return render_template('tracking.html',
                               food_log=session.get('food_log', []),
                               totals=calculate_totals(),
                               today=datetime.now().strftime('%Y-%m-%d'),
                               show_manual_entry=True,
                               manual_food_item=food_item,
                               manual_quantity=quantity)
        
        return redirect(url_for('tracking'))
    
    return render_template('tracking.html',
                         food_log=session.get('food_log', []),
                         totals=calculate_totals(),
                         today=datetime.now().strftime('%Y-%m-%d'))

def calculate_totals():
    today = datetime.now().date()
    today_log = [entry for entry in session.get('food_log', []) 
                if datetime.fromisoformat(entry['timestamp']).date() == today]
    
    return {
        'calories': sum(entry.get('calories', 0) for entry in today_log),
        'protein': sum(entry.get('protein', 0) for entry in today_log),
        'carbs': sum(entry.get('carbs', 0) for entry in today_log),
        'fats': sum(entry.get('fats', 0) for entry in today_log)
    }

@app.route('/delete_entry/<int:index>', methods=['POST'])
def delete_entry(index):
    if 'user' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    if 0 <= index < len(session['food_log']):
        session['food_log'].pop(index)
        session.modified = True
        flash('Entry deleted successfully', 'success')
    
    return redirect(url_for('tracking'))

@app.route('/ai-dietary-plan', methods=['GET', 'POST'])
def ai_dietary_plan():
    if request.method == 'POST':
        try:
            # Get and validate form data
            user_data = {
                'age': request.form['age'],
                'weight': request.form['weight'],
                'height': request.form['height'],
                'gender': request.form.get('gender', 'male'),
                'activity_level': request.form['activity_level'],
                'goal': request.form['goal'],
                'diet_preference': request.form['diet_preference'],
                'health_conditions': request.form.get('health_conditions', '')
            }

            # Generate prompt
            prompt = f"""Create a detailed 7-day personalized dietary plan as JSON with this structure:
            {{
                "day1": {{
                    "breakfast": {{
                        "menu": "...",
                        "nutrition": {{"calories": number, "protein": number, "carbs": number, "fats": number}}
                    }},
                    "lunch": {{...}},
                    "dinner": {{...}},
                    "snacks": {{...}}
                }},
                "day2": {{...}}
            }}
            
            User Profile:
            - Age: {user_data['age']}
            - Weight: {user_data['weight']}kg
            - Height: {user_data['height']}cm
            - Gender: {user_data['gender']}
            - Activity: {user_data['activity_level']}
            - Goal: {user_data['goal']}
            - Diet: {user_data['diet_preference']}
            - Health: {user_data['health_conditions'] or 'None'}
            
            Requirements:
            1. Meals should align with dietary preference
            2. Include nutritional info for each meal
            3. Provide variety across the week
            4. Consider health conditions"""
            
            # Get Gemini response
            response = model.generate_content(prompt)
            ai_diet_plan = parse_gemini_response(response.text)
            nutritional_info = calculate_nutritional_targets(user_data)
            
            return render_template('ai_dietary_plan.html',
                                ai_diet_plan=ai_diet_plan,
                                user_info=user_data,
                                nutritional_info=nutritional_info)
            
        except Exception as e:
            app.logger.error(f"Error generating plan: {str(e)}")
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('ai_dietary_plan'))
    
    # GET request - show empty form
    return render_template('ai_dietary_plan.html', ai_diet_plan=None)

if __name__ == '__main__':
    setup_database()
    app.run(host='0.0.0.0', port=5000, debug=True)