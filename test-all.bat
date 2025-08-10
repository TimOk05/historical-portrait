@echo off
chcp 65001 > nul
cls

echo 🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ИСТОРИЧЕСКОГО ПОРТРЕТА v2.0
echo ========================================================

echo.
echo 📋 Проверка файлов проекта...

REM Проверяем наличие основных файлов
if not exist "index-v2.html" (
    echo ❌ index-v2.html не найден
    goto :error
)
if not exist "index-server.html" (
    echo ❌ index-server.html не найден
    goto :error
)
if not exist "server.py" (
    echo ❌ server.py не найден
    goto :error
)
if not exist "requirements.txt" (
    echo ❌ requirements.txt не найден
    goto :error
)

echo ✅ Все основные файлы найдены

echo.
echo 🔍 Проверка Python...

REM Проверяем Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден
    goto :error
)

echo ✅ Python найден

echo.
echo 📦 Проверка зависимостей...

REM Проверяем основные зависимости
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Flask не установлен
    echo 📦 Установка зависимостей...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Ошибка установки зависимостей
        goto :error
    )
)

python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  OpenCV не установлен
    echo 📦 Установка OpenCV...
    pip install opencv-python
)

echo ✅ Зависимости готовы

echo.
echo 🌐 Тестирование сервера...

REM Запускаем сервер в фоне
echo 🚀 Запуск сервера...
start /B python server.py

REM Ждем запуска сервера
timeout /t 3 /nobreak >nul

REM Проверяем доступность сервера
curl -s http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Сервер не отвечает
    goto :error
)

echo ✅ Сервер работает

echo.
echo 📱 Тестирование клиентов...

REM Открываем клиентскую версию
echo 🖥️  Открытие клиентской версии...
start index-v2.html

REM Ждем
timeout /t 2 /nobreak >nul

REM Открываем серверную версию
echo 🌐 Открытие серверной версии...
start index-server.html

echo.
echo ✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО
echo.
echo 📊 Результаты:
echo    ✅ Все файлы найдены
echo    ✅ Python работает
echo    ✅ Зависимости установлены
echo    ✅ Сервер запущен
echo    ✅ Клиенты открыты
echo.
echo 🎯 Что тестировать:
echo    1. 📱 Отображение ролей на мобильных устройствах
echo    2. 📸 Загрузка и обработка фотографий
echo    3. 🎨 Создание исторических портретов
echo    4. 💾 Скачивание результатов
echo    5. 📤 Функция "Поделиться"
echo.
echo 🌐 Сервер доступен по адресу: http://localhost:5000
echo 📱 Клиентская версия: index-v2.html
echo 🌐 Серверная версия: index-server.html
echo.

pause
exit /b 0

:error
echo.
echo ❌ ОШИБКА ТЕСТИРОВАНИЯ
echo.
echo 🔧 Возможные решения:
echo    1. Убедитесь, что все файлы проекта на месте
echo    2. Установите Python 3.7+
echo    3. Запустите: pip install -r requirements.txt
echo    4. Проверьте, что порт 5000 свободен
echo.
pause
exit /b 1
