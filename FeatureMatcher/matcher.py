# Python Libraries
import cv2 

class FeatureMatcher:
    def __init__(self, matcher_type="FLANN"):
        # Type of feature matcher
        if matcher_type.lower() == "flann":    
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=100)   # or pass empty dictionary
            self.featureMatcher = cv2.FlannBasedMatcher(index_params,search_params)
            
        print("\n[FeaM] Setting feature matcher with {} ".format(matcher_type.upper()))
    
    def feature_matching(self, desc1, desc2, distance_threshold=0.6):
        """
        Function to match the features from two images and keep only the good ones.
        Using D. Lowes ratio to find good matches.

        INPUT
            desc1(array):               Descriptors for image 1
            desc1(array):               Descriptors for image 2
            distance_threshold(float):  Distance threhold for D Lowe's ratio test
            
        RETURN
            Return the good matches found using the Lowe's test and the mask for visulization
        """
        # Find matches with K-nn
        matches = self.featureMatcher.knnMatch(desc1, desc2, k=2)

        # Need to draw only good matches, so create a mask
        goodMatches = list()
        # ratio test as per D.Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < distance_threshold*n.distance:
                goodMatches.append([m])

        return matches, goodMatches