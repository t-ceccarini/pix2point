import cv2

def extract_and_match_features_multiple(images):
    """
    Extracts and matches features across multiple images.

    Args:
        images (list): List of input images.

    Returns:
        tuple: Keypoints, descriptors, and matches for consecutive image pairs.
    """
    orb = cv2.ORB_create()
    keypoints = []
    descriptors = []
    matches = []

    # Extract features for all images
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Match features between consecutive images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i in range(len(images) - 1):
        match = bf.match(descriptors[i], descriptors[i + 1])
        matches.append(sorted(match, key=lambda x: x.distance))

    return keypoints, matches
