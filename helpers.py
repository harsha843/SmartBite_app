def calculate_calories(user_data):
    # Basic Harris-Benedict calculation (simplified)
    if user_data['activity_level'] == 'sedentary':
        multiplier = 1.2
    elif user_data['activity_level'] == 'light':
        multiplier = 1.375
    elif user_data['activity_level'] == 'moderate':
        multiplier = 1.55
    elif user_data['activity_level'] == 'active':
        multiplier = 1.725
    else:  # extreme
        multiplier = 1.9
        
    # Simple calculation - in a real app you'd use proper formulas
    return round(2000 * multiplier)

def calculate_protein(user_data):
    # 1.6-2.2g/kg for muscle gain, 0.8g/kg otherwise
    weight = float(user_data['weight'])
    if user_data['goal'] == 'muscle_gain':
        return round(weight * 2.2, 1)
    return round(weight * 0.8, 1)

def calculate_carbs(user_data):
    # 45-65% of calories
    calories = calculate_calories(user_data)
    return round((calories * 0.5) / 4)  # 4 cal/g

def calculate_fats(user_data):
    # 20-35% of calories
    calories = calculate_calories(user_data)
    return round((calories * 0.3) / 9)  # 9 cal/g