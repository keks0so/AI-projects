import cv2

# Load the video file
cap = cv2.VideoCapture('Trim.mp4')

# Create a background subtractor object
backSub = cv2.createBackgroundSubtractorMOG2()

# Define a kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Initialize tracker
tracker = cv2.legacy.TrackerCSRT_create()

# Set the bounding box coordinates for the first frame
ret, frame = cap.read()
bbox = cv2.selectROI("Tracking", frame, False)
ok = tracker.init(frame, bbox)

# Loop through each frame in the video
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Stop the loop if the video is over
    if not ret:
        break

    # Apply background subtraction to get the foreground mask
    fgMask = backSub.apply(frame)

    # Apply morphological operations to the foreground mask
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    # Find contours in the foreground mask
    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour
    for contour in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Initialize the tracker with the bounding box coordinates
        ok = tracker.init(frame, (x, y, w, h))

    # Update the tracker
    ok, bbox = tracker.update(frame)

    # Draw the bounding box on the frame
    if ok:
        # Tracking successful
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display the frame
    cv2.imshow('Tracking', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
