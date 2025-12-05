@echo off
echo Building Tailwind CSS...
.\tailwindcss.exe -i src/input.css -o static/tailwind.css --minify
echo Tailwind CSS built successfully!
pause