# Progetto CBSD
0. remember to use 3 characters long extensions for files
1. load video in video folder
2. load tag file with same name of the video in the tag folder with extention _.tag_
3. run the following command in the terminal from project's main folder
    ```
    python detect_blinks.py --shape-predictor shapre_predictor_68_face_landmarks.dat --video video\video_name.mp4
    ```
4. verify integrity of the output in raw_data folder
5. run the following command in the terminal from project's main folder
    ```
    python preproc_svm.py --data raw_data\video_name.csv
    ```
6. run the following command instead to use the __normalized__ preprocessor
    ```
    python preproc_svm_normalizzato.py --data raw_data\video_name.csv
    ```