#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing packages
import cv2
import numpy as np


class MotionTracking(object):
    """Finds any moving object in frame."""

    def __init__(self, videoFile=None, mode='DEBUG'):
        if videoFile is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(videoFile)
        self.cap.set(propId=3, value=640)
        self.cap.set(propId=4, value=480)
        self.mode = mode

    def searchForMovingObjects(self, enableTracking=True, sensitivity=20, blurSize=10):
        """Searches for moving objects, and returns its center."""
        center = (0, 0)
        if self.cap.isOpened():
            _, frame1 = self.cap.read()
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            _, frame2 = self.cap.read()
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            difference = cv2.absdiff(gray1, gray2)

            _, threshold = cv2.threshold(
                difference, sensitivity, 255, cv2.THRESH_BINARY)

            if self.mode == "DEBUG":
                cv2.imshow('Difference', difference)
                cv2.imshow('Threshold', threshold)
            else:
                cv2.destroyWindow('Difference')
                cv2.destroyWindow('Threshold')

            threshold = cv2.blur(threshold, (blurSize, blurSize))
            _, threshold = cv2.threshold(
                threshold, sensitivity, 255, cv2.THRESH_BINARY)

            if self.mode == "RESULT":
                cv2.imshow('Result', threshold)
            else:
                cv2.destroyWindow('Result')

            if enableTracking:
                contours, hierarchy = cv2.findContours(
                    threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    movingObject = max(contours, key=cv2.contourArea)
                    cv2.drawContours(frame1, movingObject, -1, (0, 255, 0), 3)
                    M = cv2.moments(movingObject)
                    center = (int(M["m10"] / M["m00"]),
                              int(M["m01"] / M["m00"]))
                    cv2.circle(frame1, center, 3, (0, 0, 255), -1)
                    cv2.putText(frame1, "centroid", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                1)
                    cv2.putText(frame1, "(" + str(center[0]) + "," + str(center[1]) + ")", (center[0] + 10, center[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            if self.mode == "NORMAL":
                cv2.imshow('Frame', frame1)
            else:
                cv2.destroyWindow('Frame')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()

            return center


# Example of usage
if __name__ == "__main__":

    motionTracker = MotionTracking(mode='NORMAL')

    while True:
        center = motionTracker.searchForMovingObjects(
            enableTracking=True, sensitivity=20, blurSize=10)
        print(center)
