@echo off
chcp 65001 > nul
cls

echo 🚀 ЗАПУСК СЕРВЕРА ИСТОРИЧЕСКИХ ПОРТРЕТОВ
echo =========================================

echo.
echo 📋 Проверка зависимостей...

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python 3.7+
    pause
    exit /b 1
)

echo ✅ Python найден

REM Проверяем наличие pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip не найден! Установите pip
    pause
    exit /b 1
)

echo ✅ pip найден

echo.
echo 📦 Установка зависимостей...

REM Устанавливаем зависимости
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Ошибка установки зависимостей
    pause
    exit /b 1
)

echo ✅ Зависимости установлены

echo.
echo 🌐 Запуск сервера...
echo 📍 Сервер будет доступен по адресу: http://localhost:5000
echo 📱 Клиент: http://localhost:5000/index-server.html
echo.

REM Запускаем сервер
python server.py

pause
