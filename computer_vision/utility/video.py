import sys
import cv2
import numpy as np
import argparse

'''
Decorator pattern example for a camera and different filters

ref: https://refactoring.guru/design-patterns/decorator/python/example
'''

class VideoCapture():
    def __init__(self) -> None:
        super().__init__()
        self._video_capture = None

    def set_source(self, source) -> int:
        # Get a reference to webcam
        try:
            # input from IP camera
            if ':' in source:
                self._video_capture = cv2.VideoCapture(source)
            # input from webcam
            elif source.isdigit():
                self._video_capture = cv2.VideoCapture(int(source))
            # input from video
            else:
                self._video_capture = cv2.VideoCapture(source)
        except:
            return False

        return True

    def read(self) -> (bool, np.ndarray):
        # Grab a single frame of video
        return self._video_capture.read()

    def release(self):
        return self._video_capture.release()

class ImageFilter(VideoCapture):
    _video_capture: VideoCapture = None

    def __init__(self, video_capture: VideoCapture) -> None:
        super().__init__()
        self._video_capture = video_capture

    @property
    def video_capture(self):
        return self._video_capture

    def read(self):
        return self._video_capture.read()

    def __del__(self):
        return self._video_capture.release()

class FishEye(ImageFilter):

    def set(self, value):
        # Fisheye distortion correction for HIKVISION Camera
        self.__rfd_matrix = np.eye(3, dtype = np.float32)
        self.__rfd_dist_coeff = np.zeros((4,1),np.float64)

        # negative to remove barrel distortion
        self.__rfd_dist_coeff[0,0] = 0.0

        self.__rfd_dist_coeff[1,0] = 0.0
        self.__rfd_dist_coeff[2,0] = 0.0
        self.__rfd_dist_coeff[3,0] = 0.0

        self.__rfd_dist_coeff[0,0] = value

    def read(self):
        # Grab a single frame of video
        ret, frame = self.video_capture.read()

        if ret:
            frame = cv2.undistort(frame, self.__rfd_matrix, self.__rfd_dist_coeff)
            return ret, frame
        else:
            return ret, None

class Histogram(ImageFilter):
    def read(self):
        # Grab a single frame of video
        ret, frame = self.video_capture.read()

        if ret:
            img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            # equalize the histogram of the Y channel
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            return ret, frame
        else:
            return ret, None

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-vs", "--video_source", required=True,
        help="video source - integrated webcam, video filepath or URL")
    ap.add_argument("-fe", "--fisheye", type=float,
        help="parameters to adjust fisheye")
    ap.add_argument("-hg", "--histogram", type=bool,
        help="parameters to adjust fisheye")
    args = vars(ap.parse_args())

    vc = VideoCapture()
    success = vc.set_source(args["video_source"])

    if success:
        if args["fisheye"]:
            vc = FishEye(vc)
            vc.set(value=args["fisheye"])

        if args["histogram"]:
            vc = Histogram(vc)

        while True:
            ret, frame = vc.read()

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
    else:
        print("failed to set video source")
        exit()