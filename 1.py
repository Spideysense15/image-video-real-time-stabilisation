import cv2
import numpy as np

def stabilize_video_from_camera():
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Feature detector
    orb = cv2.ORB_create()

    # Feature matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        # Match descriptors
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        if H is not None:
            # Warp current frame
            height, width, _ = prev_frame.shape
            stabilized_frame = cv2.warpPerspective(curr_frame, H, (width, height))
        else:
            stabilized_frame = curr_frame

        # Show the stabilized frame
        cv2.imshow('Stabilized Frame', stabilized_frame)

        # Update previous frame
        prev_gray = curr_gray
        prev_frame = curr_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the stabilization
stabilize_video_from_camera()
