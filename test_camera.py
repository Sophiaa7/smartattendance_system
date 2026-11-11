import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera could not be opened!")
    print("   - Close any other app using the camera (Zoom, Teams, etc.)")
    print("   - Check Windows Camera Privacy settings")
else:
    print("✅ Camera opened successfully!")

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera Test - Press Q to quit", frame)
    else:
        print("Failed to read frame")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()