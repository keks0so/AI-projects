import cv2

cap = cv2.VideoCapture("Trim.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break



    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Apply thresholding to clean up the mask
    h, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours in the thresholded mask
    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow('frame', frame)
    cv2.imshow('mask', fgmask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
