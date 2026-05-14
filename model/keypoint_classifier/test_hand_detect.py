import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 🟩 UPDATE THIS PATH BELOW — USE A REAL IMAGE FROM YOUR DATASET
image_path = r"C:\SignAI-Sign-Language-Recognition-and-Translation-main\SignAI\dataset\ProcessedData_vivit\beautiful\MVI_9569.MOV"
# Check if file exists before reading
if not os.path.exists(image_path):
    print(f"❌ File not found: {image_path}")
else:
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load the image. Check the file format or path.")
    else:
        print("✅ Image loaded successfully.")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            print("✅ Hand detected!")
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Hand Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ No hand detected in the image.")
