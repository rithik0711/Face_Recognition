import cv2

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the default webcam (0)
webcam = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Loop to continuously get frames
while True:
    # Read frame from webcam
    ret, img = webcam.read()
    if not ret:
        print("Error: Failed to read from webcam.")
        break

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the output
    cv2.imshow("Face Detection", img)

    # Press ESC key to exit
    if cv2.waitKey(10) == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
