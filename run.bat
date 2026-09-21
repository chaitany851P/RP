@echo off
setlocal
echo ========================================================
echo       Industrial Safety Detection System (RP)
echo ========================================================
echo 1. Run All Tests (test_all.py)
echo 2. Run Full Pipeline on video.mp4
echo 3. Run Full Pipeline on Webcam
echo 4. Run Zone Setup Tool
echo 5. Run Fall Detector
echo 6. Run Loitering Detector
echo 7. Run Prolonged Exposure Detector
echo 8. Run Unauthorized Entry Detector
echo 9. Run Unsafe Posture Detector
echo ========================================================
set /p choice="Choose an option (1-9) [Default: 2]: "
if "%choice%"=="" set choice=2

if "%choice%"=="1" (
    py test_all.py
) else if "%choice%"=="2" (
    py core\pipeline.py video.mp4
) else if "%choice%"=="3" (
    py core\pipeline.py 0
) else if "%choice%"=="4" (
    py zone_setup.py video.mp4
) else if "%choice%"=="5" (
    py fall_detection\fall_detector.py video.mp4
) else if "%choice%"=="6" (
    py loitering_detection\loitering_detector.py video.mp4
) else if "%choice%"=="7" (
    py prolonged_exposure\prolonged_exposure_detector.py video.mp4
) else if "%choice%"=="8" (
    py unauthorized_entry\unauthorized_entry_detector.py video.mp4
) else if "%choice%"=="9" (
    py unsafe_posture\unsafe_posture_detector.py video.mp4
) else (
    echo Invalid choice.
)
pause
