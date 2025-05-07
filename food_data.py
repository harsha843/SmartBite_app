# Organized Indian food items by meal category
food_categories = {
    'breakfast': [
        'poha', 'upma', 'idli', 'dosa', 'vada', 'pongal', 'paratha', 'aloo_paratha',
        'chapati', 'missi_roti', 'aloo_tikki', 'kachori', 'daal_puri', 'poha'
    ],
    'lunch': [
        'biryani', 'pulao', 'khichdi', 'rajma_chawal', 'dal_chawal', 'sambar_rice', 
        'curd_rice', 'aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_shimla_mirch',
        'bhindi_masala', 'chana_masala', 'dal_makhani', 'dal_tadka', 'kadai_paneer',
        'palak_paneer', 'paneer_butter_masala', 'navrattan_korma', 'kofta', 'dum_aloo',
        'karela_bharta', 'makki_di_roti_sarson_da_saag', 'litti_chokha'
    ],
    'dinner': [
        'roti', 'naan', 'kulcha', 'paratha', 'missi_roti', 'dal_makhani', 'dal_tadka',
        'paneer_butter_masala', 'kadai_paneer', 'palak_paneer', 'navrattan_korma',
        'butter_chicken', 'chicken_tikka_masala', 'chicken_razala', 'maach_jhol',
        'kofta', 'dum_aloo', 'bhindi_masala', 'aloo_gobi', 'aloo_matar'
    ],
    'snacks': [
        'samosa', 'kachori', 'pakora', 'vada', 'dahi_vada', 'bhel_puri', 'pani_puri',
        'sevpuri', 'aloo_tikki', 'pav_bhaji', 'vada_pav', 'dosa', 'idli', 'uttapam',
        'kuzhi_paniyaram', 'bhajji', 'bread_pakoda', 'mirchi_bajji', 'poha'
    ],
    'desserts': [
        'gulab_jamun', 'rasgulla', 'rasmalai', 'jalebi', 'barfi', 'halwa', 'gajar_ka_halwa',
        'sooji_halwa', 'kheer', 'phirni', 'rabri', 'shrikhand', 'sandesh', 'ladoo',
        'besan_ladoo', 'motichoor_ladoo', 'peda', 'kalakand', 'mysore_pak', 'malpua',
        'imarti', 'cham_cham', 'double_ka_meetha', 'basundi', 'doodhpak', 'sheer_korma',
        'qubani_ka_meetha', 'adhirasam', 'ariselu', 'bandar_laddu', 'chikki', 'dharwad_pedha',
        'gavvalu', 'ghevar', 'kajjikaya', 'kakinada_khaja', 'ledikeni', 'lyangcha', 'modak',
        'pithe', 'poornalu', 'pootharekulu', 'shankarpali', 'sheera', 'sohan_halwa',
        'sohan_papdi', 'sutar_feni', 'unni_appam'
    ]
}

# Nutrition data for Indian food items organized by category
nutrition_data = {
    'breakfast': {
        'poha': {'calories': 250, 'carbs': 40, 'protein': 6, 'fats': 8, 'vitamins': 5},
        'chapati': {'calories': 120, 'carbs': 25, 'protein': 4, 'fats': 2, 'vitamins': 2},
        'aloo_tikki': {'calories': 200, 'carbs': 30, 'protein': 6, 'fats': 8, 'vitamins': 4},
        'kachori': {'calories': 300, 'carbs': 40, 'protein': 7, 'fats': 15, 'vitamins': 4},
        'daal_puri': {'calories': 280, 'carbs': 35, 'protein': 10, 'fats': 10, 'vitamins': 4},
        'missi_roti': {'calories': 200, 'carbs': 35, 'protein': 7, 'fats': 6, 'vitamins': 4}
    },
    'lunch': {
        'biryani': {'calories': 350, 'carbs': 45, 'protein': 15, 'fats': 12, 'vitamins': 4},
        'aloo_gobi': {'calories': 120, 'carbs': 18, 'protein': 4, 'fats': 5, 'vitamins': 6},
        'aloo_matar': {'calories': 160, 'carbs': 25, 'protein': 6, 'fats': 6, 'vitamins': 5},
        'bhindi_masala': {'calories': 110, 'carbs': 15, 'protein': 4, 'fats': 5, 'vitamins': 7},
        'chana_masala': {'calories': 180, 'carbs': 25, 'protein': 9, 'fats': 6, 'vitamins': 7},
        'dal_makhani': {'calories': 350, 'carbs': 30, 'protein': 12, 'fats': 20, 'vitamins': 6},
        'kadai_paneer': {'calories': 350, 'carbs': 12, 'protein': 18, 'fats': 25, 'vitamins': 7},
        'palak_paneer': {'calories': 220, 'carbs': 15, 'protein': 14, 'fats': 12, 'vitamins': 9},
        'paneer_butter_masala': {'calories': 330, 'carbs': 18, 'protein': 16, 'fats': 22, 'vitamins': 6},
        'dum_aloo': {'calories': 220, 'carbs': 25, 'protein': 4, 'fats': 12, 'vitamins': 5},
        'karela_bharta': {'calories': 100, 'carbs': 15, 'protein': 3, 'fats': 4, 'vitamins': 9},
        'makki_di_roti_sarson_da_saag': {'calories': 300, 'carbs': 40, 'protein': 10, 'fats': 12, 'vitamins': 8}
    },
    'dinner': {
        'naan': {'calories': 290, 'carbs': 50, 'protein': 9, 'fats': 10, 'vitamins': 2},
        'butter_chicken': {'calories': 450, 'carbs': 10, 'protein': 25, 'fats': 30, 'vitamins': 5},
        'chicken_tikka_masala': {'calories': 380, 'carbs': 15, 'protein': 25, 'fats': 25, 'vitamins': 5},
        'dal_makhani': {'calories': 350, 'carbs': 30, 'protein': 12, 'fats': 20, 'vitamins': 6},
        'kadai_paneer': {'calories': 350, 'carbs': 12, 'protein': 18, 'fats': 25, 'vitamins': 7},
        'palak_paneer': {'calories': 220, 'carbs': 15, 'protein': 14, 'fats': 12, 'vitamins': 9},
        'bhindi_masala': {'calories': 110, 'carbs': 15, 'protein': 4, 'fats': 5, 'vitamins': 7},
        'maach_jhol': {'calories': 250, 'carbs': 10, 'protein': 22, 'fats': 15, 'vitamins': 6}
    },
    'snacks': {
        'aloo_tikki': {'calories': 200, 'carbs': 30, 'protein': 6, 'fats': 8, 'vitamins': 4},
        'kachori': {'calories': 300, 'carbs': 40, 'protein': 7, 'fats': 15, 'vitamins': 4},
        'kuzhi_paniyaram': {'calories': 200, 'carbs': 30, 'protein': 5, 'fats': 8, 'vitamins': 4},
        'poha': {'calories': 250, 'carbs': 40, 'protein': 6, 'fats': 8, 'vitamins': 5}
    },
    'desserts': {
        'gulab_jamun': {'calories': 380, 'carbs': 65, 'protein': 5, 'fats': 15, 'vitamins': 2},
        'rasgulla': {'calories': 300, 'carbs': 60, 'protein': 6, 'fats': 8, 'vitamins': 3},
        'rasmalai': {'calories': 320, 'carbs': 40, 'protein': 8, 'fats': 15, 'vitamins': 5},
        'jalebi': {'calories': 400, 'carbs': 80, 'protein': 3, 'fats': 12, 'vitamins': 1},
        'gajar_ka_halwa': {'calories': 350, 'carbs': 50, 'protein': 5, 'fats': 15, 'vitamins': 8},
        'phirni': {'calories': 280, 'carbs': 35, 'protein': 7, 'fats': 10, 'vitamins': 4},
        'basundi': {'calories': 280, 'carbs': 35, 'protein': 8, 'fats': 12, 'vitamins': 5},
        'shrikhand': {'calories': 330, 'carbs': 45, 'protein': 7, 'fats': 15, 'vitamins': 4},
        'sandesh': {'calories': 250, 'carbs': 30, 'protein': 8, 'fats': 10, 'vitamins': 4},
        'kalakand': {'calories': 350, 'carbs': 40, 'protein': 8, 'fats': 20, 'vitamins': 4},
        'mysore_pak': {'calories': 450, 'carbs': 60, 'protein': 5, 'fats': 25, 'vitamins': 2},
        'double_ka_meetha': {'calories': 400, 'carbs': 60, 'protein': 6, 'fats': 15, 'vitamins': 3}
    }
}

# Create a flattened version for easy access
all_nutrition_data = {}
for category, foods in nutrition_data.items():
    for food, nutrients in foods.items():
        all_nutrition_data[food] = nutrients

def get_nutrition_data(food_name):
    return all_nutrition_data.get(food_name)

def get_foods_by_category(category):
    return nutrition_data.get(category, {})