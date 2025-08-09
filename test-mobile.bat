@echo off
chcp 65001 > nul
cls

echo 📱 ТЕСТИРОВАНИЕ НА МОБИЛЬНЫХ УСТРОЙСТВАХ
echo ======================================

echo.
echo 🔧 Открываю тестовую страницу...
start test-mobile-roles.html

echo.
echo 📋 Инструкции для тестирования:
echo.
echo 1. 📱 Откройте страницу на телефоне
echo 2. 🔍 Проверьте, отображаются ли роли
echo 3. 👆 Попробуйте выбрать разные роли
echo 4. 📊 Посмотрите информацию об устройстве
echo.
echo 🌐 Или откройте в браузере:
echo    test-mobile-roles.html
echo.
echo 📱 Для тестирования на телефоне:
echo    1. Запустите локальный сервер:
echo       python -m http.server 8080
echo    2. Откройте на телефоне:
echo       http://[IP-адрес]:8080/test-mobile-roles.html
echo.

pause
