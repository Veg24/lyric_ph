import cv2
import mediapipe as mp
import os

def get_next_take_filename():
    """
    Finds the next available filename in the format 'take_X.mp4'
    to prevent overwriting existing recordings.
    """
    index = 1
    while True:
        filename = f"take_{index}.mp4"
        if not os.path.exists(filename):
            return filename
        index += 1

def is_v_fingers_raised(hand_landmarks):
    """
    Checks if a hand has its index and middle fingers raised (the V fingers).
    Ring and pinky fingers are ignored to ensure maximum detection responsiveness 
    and tilt tolerance.
    
    Note: In screen coordinates, the Y-axis increases downwards, 
    so "above" means a smaller Y value.
    """
    landmarks = hand_landmarks.landmark
    
    # Check if index finger is raised (TIP is above MCP)
    index_open = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y
    
    # Check if middle finger is raised (TIP is above MCP)
    middle_open = landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y
    
    return index_open and middle_open

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    
    # Configure Hand Tracking for high responsiveness
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Set camera buffer size to 1 to eliminate frame caching lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set screen resolution to a compact size (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get actual camera resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get camera FPS (fallback to 20.0 if unable to read from camera properties)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 20.0

    # Define the codec and create VideoWriter object to save the recording with automatic name incrementing
    output_filename = get_next_take_filename()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (actual_width, actual_height))

    print("=== Hand Tracking & V-Fingers Blur with Recording ===")
    print(f"Camera Resolution: {actual_width}x{actual_height} @ {fps} FPS")
    print(f"Recording status: Writing video to '{output_filename}'...")
    print("Instructions:")
    print("1. Raise the index and middle fingers on AT LEAST 1 hand (V pose) to trigger the blur effect.")
    print("2. Press 'q' on the video window to stop recording and exit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break
            
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        v_hands_count = 0
        
        # Count hands showing the raised V-fingers
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                if is_v_fingers_raised(hand_lms):
                    v_hands_count += 1
        
        # Copy frame for output display
        output_frame = frame.copy()
        
        # Apply Gaussian Blur if at least 1 hand shows the raised index and middle fingers
        if v_hands_count >= 1:
            # Optimize Blur performance (Downscale -> Blur -> Upscale)
            small_frame = cv2.resize(output_frame, (w // 4, h // 4))
            small_blurred = cv2.GaussianBlur(small_frame, (15, 15), 0)
            output_frame = cv2.resize(small_blurred, (w, h), interpolation=cv2.INTER_LINEAR)
            
        # Write the processed frame to the video file
        out.write(output_frame)
        
        # Display the output (without any skeleton lines or text overlays)
        cv2.imshow("Hand Tracking - V-Fingers Blur", output_frame)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\nRecording stopped. Video successfully saved to '{output_filename}' in the project directory.")

if __name__ == "__main__":
    main()
