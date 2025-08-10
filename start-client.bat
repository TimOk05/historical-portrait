@echo off
chcp 65001 > nul
cls

echo 🎭 ЗАПУСК КЛИЕНТА ИСТОРИЧЕСКИХ ПОРТРЕТОВ
echo ========================================

echo.
echo 📋 Доступные версии:
echo.
echo 1. 🖥️  Клиентская версия (index-v2.html)
echo    - Работает полностью в браузере
echo    - Не требует сервера
echo    - Быстрая обработка
echo.
echo 2. 🌐 Серверная версия (index-server.html)
echo    - Требует запущенный сервер
echo    - Лучшее качество обработки
echo    - Более продвинутые эффекты
echo.

set /p choice="Выберите версию (1 или 2): "

if "%choice%"=="1" (
    echo.
    echo 🖥️ Запуск клиентской версии...
    start index-v2.html
) else if "%choice%"=="2" (
    echo.
    echo 🌐 Запуск серверной версии...
    echo ⚠️  Убедитесь, что сервер запущен!
    echo.
    start index-server.html
) else (
    echo.
    echo ❌ Неверный выбор!
    pause
    exit /b 1
)

echo.
echo ✅ Приложение запущено!
echo 📱 Откройте в браузере для использования
echo.

pause
