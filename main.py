from FRSD import do_frst
import cv2
import numpy as np
import dlib
import sys
import argparse
import imutils

from imutils import face_utils

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

ym = 5
xm = 5


def extract_roi(image, shape, i, j):
    # extract the ROI of the face region as a separate image
    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = image[y-ym:y+h+ym,x-xm:x+w+xm]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    return roi


while True:
    # Capture frame-by-frame
    ret, image = cap.read()

    # Resize the image

    image = imutils.resize(image, width=500)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect face
    rects = detector(gray, 1)
   # gray = do_frst(gray, 20, 10, 2, 3)   tried to add fsrd to main image but it will have errors regarding image type
    # print image
    cv2.imshow("webcam", gray)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (ri, rj) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        (li, lj) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']

        cv2.imshow("right eye", (extract_roi(image, shape, ri, rj)))
        cv2.imshow("left eye", (extract_roi(image, shape, li, lj)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()