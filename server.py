#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
import json
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех доменов

# Конфигурация
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Создаем папки если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Исторические роли и их эффекты
HISTORICAL_ROLES = {
    'peasant': {
        'name': 'Крестьянин',
        'background_colors': [(143, 188, 143), (107, 142, 35), (85, 107, 47)],
        'effects': {'sepia': 0.3, 'contrast': 1.1, 'brightness': 1.0}
    },
    'hussar': {
        'name': 'Крылатый гусар',
        'background_colors': [(139, 0, 0), (101, 67, 33), (47, 27, 20)],
        'effects': {'sepia': 0.2, 'contrast': 1.3, 'brightness': 1.1}
    },
    'boyar': {
        'name': 'Боярин',
        'background_colors': [(75, 0, 130), (48, 25, 52), (30, 15, 33)],
        'effects': {'sepia': 0.4, 'contrast': 1.2, 'brightness': 0.9}
    },
    'prince': {
        'name': 'Князь',
        'background_colors': [(255, 215, 0), (184, 134, 11), (139, 115, 85)],
        'effects': {'sepia': 0.1, 'contrast': 1.4, 'brightness': 1.2}
    },
    'merchant': {
        'name': 'Купец',
        'background_colors': [(34, 139, 34), (30, 94, 30), (20, 52, 20)],
        'effects': {'sepia': 0.3, 'contrast': 1.1, 'brightness': 1.0}
    },
    'monk': {
        'name': 'Монах',
        'background_colors': [(105, 105, 105), (74, 74, 74), (47, 47, 47)],
        'effects': {'sepia': 0.5, 'contrast': 0.9, 'brightness': 0.8}
    }
}

def allowed_file(filename):
    """Проверяем, что файл имеет разрешенное расширение"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_historical_background(width, height, role_config):
    """Создаем исторический фон"""
    # Создаем градиентный фон
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    colors = role_config['background_colors']
    for i in range(height):
        # Интерполируем цвета для создания градиента
        ratio = i / height
        if ratio <= 0.5:
            # Первая половина - градиент между первым и вторым цветом
            local_ratio = ratio * 2
            color = np.array(colors[0]) * (1 - local_ratio) + np.array(colors[1]) * local_ratio
        else:
            # Вторая половина - градиент между вторым и третьим цветом
            local_ratio = (ratio - 0.5) * 2
            color = np.array(colors[1]) * (1 - local_ratio) + np.array(colors[2]) * local_ratio
        
        background[i, :] = color
    
    # Добавляем текстуру
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    background = cv2.addWeighted(background, 0.9, noise, 0.1, 0)
    
    return background

def apply_historical_effects(image, role_config):
    """Применяем исторические эффекты к изображению"""
    effects = role_config['effects']
    
    # Конвертируем в PIL для лучшей обработки
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Применяем сепию
    if effects.get('sepia', 0) > 0:
        pil_image = apply_sepia_filter(pil_image, effects['sepia'])
    
    # Применяем контраст
    if effects.get('contrast', 1) != 1:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(effects['contrast'])
    
    # Применяем яркость
    if effects.get('brightness', 1) != 1:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(effects['brightness'])
    
    # Конвертируем обратно в OpenCV формат
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def apply_sepia_filter(image, intensity):
    """Применяем сепия фильтр"""
    # Создаем сепия матрицу
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    # Применяем сепия эффект
    sepia_image = image.convert('RGB')
    sepia_image = sepia_image.convert('RGB', matrix=sepia_matrix)
    
    # Смешиваем с оригиналом
    if intensity < 1:
        sepia_image = Image.blend(image, sepia_image, intensity)
    
    return sepia_image

def extract_foreground(image):
    """Извлекаем передний план (человека) из изображения"""
    # Конвертируем в HSV для лучшего выделения
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Создаем маску для кожи (приблизительно)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Расширяем маску для захвата большего количества пикселей
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    
    # Применяем размытие для сглаживания краев
    skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)
    
    # Создаем альфа-канал
    alpha = skin_mask.astype(np.float32) / 255.0
    
    return alpha

def create_historical_portrait(user_image, role_id):
    """Создаем исторический портрет"""
    if role_id not in HISTORICAL_ROLES:
        raise ValueError(f"Неизвестная роль: {role_id}")
    
    role_config = HISTORICAL_ROLES[role_id]
    
    # Размеры портрета
    portrait_width = 600
    portrait_height = 800
    
    # Создаем фон
    background = create_historical_background(portrait_width, portrait_height, role_config)
    
    # Обрабатываем пользовательское изображение
    # Изменяем размер для вписывания в портрет
    user_height, user_width = user_image.shape[:2]
    max_user_width = int(portrait_width * 0.6)
    max_user_height = int(portrait_height * 0.5)
    
    scale = min(max_user_width / user_width, max_user_height / user_height)
    new_width = int(user_width * scale)
    new_height = int(user_height * scale)
    
    resized_user = cv2.resize(user_image, (new_width, new_height))
    
    # Извлекаем передний план
    alpha = extract_foreground(resized_user)
    
    # Позиционируем пользовательское изображение
    x_offset = (portrait_width - new_width) // 2
    y_offset = int(portrait_height * 0.2)
    
    # Накладываем пользовательское изображение на фон
    for y in range(new_height):
        for x in range(new_width):
            if y + y_offset < portrait_height and x + x_offset < portrait_width:
                alpha_val = alpha[y, x]
                if alpha_val > 0.1:  # Порог для видимости
                    background[y + y_offset, x + x_offset] = (
                        resized_user[y, x] * alpha_val + 
                        background[y + y_offset, x + x_offset] * (1 - alpha_val)
                    ).astype(np.uint8)
    
    # Применяем исторические эффекты
    result = apply_historical_effects(background, role_config)
    
    # Добавляем текст и рамку
    result = add_decorations(result, role_config)
    
    return result
    
    # Применяем исторические эффекты
    result = apply_historical_effects(background, role_config)
    
    # Добавляем текст и рамку
    result = add_decorations(result, role_config)
    
    return result

def add_decorations(image, role_config):
    """Добавляем декорации к портрету"""
    height, width = image.shape[:2]
    
    # Добавляем заголовок
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    # Белый цвет для текста
    text_color = (255, 255, 255)
    
    # Название роли
    text = role_config['name'].upper()
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = 60
    
    # Добавляем тень
    cv2.putText(image, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Подпись
    subtitle = "XVI век • Речь Посполитая"
    subtitle_scale = 0.8
    subtitle_thickness = 2
    subtitle_size = cv2.getTextSize(subtitle, font, subtitle_scale, subtitle_thickness)[0]
    subtitle_x = (width - subtitle_size[0]) // 2
    subtitle_y = height - 30
    
    cv2.putText(image, subtitle, (subtitle_x, subtitle_y), font, subtitle_scale, (184, 134, 11), subtitle_thickness)
    
    # Добавляем рамку
    border_color = (255, 215, 0) if role_config['name'] == 'Князь' else (184, 134, 11)
    border_thickness = 3 if role_config['name'] == 'Князь' else 2
    cv2.rectangle(image, (10, 10), (width - 10, height - 10), border_color, border_thickness)
    
    return image

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/generate', methods=['POST'])
def generate_portrait():
    """Генерируем исторический портрет"""
    try:
        # Проверяем наличие файла
        if 'image' not in request.files:
            return jsonify({'error': 'Файл не найден'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
        
        # Проверяем размер файла
        file.seek(0, 2)  # Перемещаемся в конец файла
        file_size = file.tell()
        file.seek(0)  # Возвращаемся в начало
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'Файл слишком большой'}), 400
        
        # Получаем роль
        role_id = request.form.get('role', 'peasant')
        if role_id not in HISTORICAL_ROLES:
            return jsonify({'error': 'Неизвестная роль'}), 400
        
        # Читаем изображение
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        user_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if user_image is None:
            return jsonify({'error': 'Не удалось прочитать изображение'}), 400
        
        logger.info(f"Обрабатываем изображение для роли: {role_id}")
        
        # Создаем портрет
        result_image = create_historical_portrait(user_image, role_id)
        
        # Конвертируем в base64
        _, buffer = cv2.imencode('.png', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'role': role_id,
            'role_name': HISTORICAL_ROLES[role_id]['name']
        })
        
    except Exception as e:
        logger.error(f"Ошибка при генерации портрета: {str(e)}")
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500

@app.route('/roles', methods=['GET'])
def get_roles():
    """Получаем список доступных ролей"""
    roles = []
    for role_id, config in HISTORICAL_ROLES.items():
        roles.append({
            'id': role_id,
            'name': config['name']
        })
    
    return jsonify({'roles': roles})

@app.route('/')
def index():
    """Главная страница"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Исторический портрет - API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>⚜️ Исторический портрет - API</h1>
        <p>Сервер для создания исторических портретов</p>
        
        <h2>Доступные эндпоинты:</h2>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Проверка здоровья сервера</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /roles</h3>
            <p>Получить список доступных ролей</p>
        </div>
        
        <div class="endpoint">
            <h3>POST /generate</h3>
            <p>Создать исторический портрет</p>
            <p><strong>Параметры:</strong></p>
            <ul>
                <li><code>image</code> - файл изображения (JPG, PNG, WebP)</li>
                <li><code>role</code> - ID роли (peasant, hussar, boyar, prince, merchant, monk)</li>
            </ul>
        </div>
        
        <h2>Пример использования:</h2>
        <pre><code>
curl -X POST -F "image=@photo.jpg" -F "role=prince" http://localhost:5000/generate
        </code></pre>
    </body>
    </html>
    '''

if __name__ == '__main__':
    logger.info("Запуск сервера исторических портретов...")
    logger.info("Доступные роли: " + ", ".join(HISTORICAL_ROLES.keys()))
    
    # Запускаем сервер
    app.run(host='0.0.0.0', port=5000, debug=True)
