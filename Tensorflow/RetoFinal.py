import cv2
import sys
video_capture = cv2.VideoCapture(0)

while True:
    #Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # recorteColor = frame[180:300, 256:384]
    # recorteGray = gray[180:300, 256:384]

    recorteColor = frame[220:268, 296:344]
    recorteGray = gray[220:268, 296:344]

    cv2.imshow('Video', recorteColor)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()