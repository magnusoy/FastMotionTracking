#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fast_motion_tracking import MotionTracking

if __name__ == "__main__":

    motionTracker = MotionTracking(mode='NORMAL')

    while True:
        center = motionTracker.searchForMovingObjects(
            enableTracking=True, sensitivity=20, blurSize=10)
        print(center)
