import cv2
import mediapipe as mp

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to detect SOS gesture
def is_sos_gesture(hand_landmarks):
    # Extract the landmarks for fingers (0: wrist, 4: thumb_tip, 8: index_tip, 12: middle_tip, 16: ring_tip, 20: pinky_tip)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Gesture 1 (S): Raise index, middle, and ring fingers (pinky and thumb should be down)
    if (index_tip.y < thumb_tip.y and
        middle_tip.y < ring_tip.y and
        ring_tip.y < pinky_tip.y and 
        thumb_tip.y > index_tip.y and
        pinky_tip.y > ring_tip.y):
        return "S"

    # Gesture 2 (O): Create a circle between thumb and index finger
    thumb_index_dist = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
    if thumb_index_dist < 0.05:  # Customize this threshold based on your scale
        return "O"

    return None

# Initialize video capture (0 for webcam or provide a video file path)
cap = cv2.VideoCapture(0)

# Initialize the Hands model
with mp_hands.Hands(
    static_image_mode=False,      # Detect hands in real-time
    max_num_hands=1,              # Max number of hands to track
    min_detection_confidence=0.7, # Detection confidence threshold
    min_tracking_confidence=0.5   # Tracking confidence threshold
) as hands:

    sos_sequence = []
    sos_triggered = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame to detect hands
        result = hands.process(rgb_frame)

        # Draw hand landmarks if detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect the hand gesture
                gesture = is_sos_gesture(hand_landmarks)

                if gesture:
                    sos_sequence.append(gesture)
                    print(f"Detected gesture: {gesture}")

                    # Only keep track of the last 3 gestures
                    if len(sos_sequence) > 3:
                        sos_sequence.pop(0)

                    # Check if the sequence matches 'SOS'
                    if sos_sequence == ['S', 'O', 'S']:
                        sos_triggered = True

        # Display the frame
        cv2.imshow('SOS Gesture Detection', frame)

        # Trigger an SOS alert
        if sos_triggered:
            print("SOS Alert Triggered!")
            # You can add code to send an actual alert here
            sos_triggered = False
            sos_sequence = []  # Reset sequence after triggering

        # Exit loop on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()