import cv2
import numpy as np

cap_right = cv2.VideoCapture("udpsrc port=5200 !  application/x-rtp, encoding-name=JPEG,"
                             "payload=26 ! rtpjpegdepay !  jpegdec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

cap_left = cv2.VideoCapture("udpsrc port=5201 !  application/x-rtp, encoding-name=JPEG,"
                            "payload=26 ! rtpjpegdepay !  jpegdec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)


print(f"Image Capture Right : {cap_right.isOpened()}")
print(f"Image Capture Left : {cap_left.isOpened()}")


def detect_pupil(img, title):
    img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)

    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thV, imgT = cv2.threshold(imgG, 35, 255, cv2.THRESH_BINARY)
    highTH = thV
    lowTH = thV / 2
    # cv2.imshow('Tresh', imgT)

    imgB = cv2.medianBlur(imgT, 7)
    # cv2.imshow("Blur", imgB)

    # Find the binary image with edges from the thresholded image
    imgE = cv2.Canny(imgB, threshold1=lowTH, threshold2=highTH)
    # cv2.imshow('Canny' + title, imgE)

    # Process the image for circles using the Hough transform
    circles = cv2.HoughCircles(imgE, cv2.HOUGH_GRADIENT, 2, img.shape[0] / 64, param1=24, param2=62,
                               minRadius=15, maxRadius=35)

    # Determine if any circles were found
    if circles is not None:
        if len(circles) > 1:
            print("False circles detected!")
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # draw the circles
        x, y, r = circles[0]

        cv2.circle(img, (x, y), r, (0, 250, 0), 2)

        cv2.line(img, (x, 0), (x, 600), (0, 255, 0), 1)
        cv2.line(img, (0, y), (600, y), (0, 255, 0), 1)

    cv2.imshow(title, img)


while True:
    if cap_right.grab():
        flag, img = cap_right.read()
        if not flag:
            continue
        else:
            detect_pupil(img, "right")

    if cap_left.grab():
        flag, img = cap_left.read()
        if not flag:
            continue
        else:
            detect_pupil(img, "left")


