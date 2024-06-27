import cv2

def show_camera_feed(camera_index):
    # Open the video capture with the specified index
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to show the camera feed
show_camera_feed(2)
