@ECHO OFF
title Sex Machine
set /p _inputname= Please load your video in the "video" folder and input the name:
python video_to_prevision.py -p shape_predictor_68_face_landmarks.dat -v video\%_inputname%
@ECHO ON
@ECHO OFF
python .\show_video_EAR_NOFACE.py -v .\video\%_inputname% -q .\prevision\data_final_%_inputname:~0,-4%.csv
echo Your video has been processed.
set /p DUMMY= Hit enter to continue...
