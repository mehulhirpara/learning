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
        self._height = 0
        self._width = 0

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

            self._height = int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._width = int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        except:
            return False

        return True

    def size(self):
        return (self._width, self._height)

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

    def size(self):
        return self._video_capture.size()

    def read(self):
        return self._video_capture.read()

    def __del__(self):
        return self._video_capture.release()

class FishEye(ImageFilter):

    def set(self, value):
        # Fisheye distortion correction for HIKVISION Camera
        self.__rfd_dist_coeff = np.zeros((4,1), np.float64)

        # negative to remove barrel distortion
        self.__rfd_dist_coeff[0,0] = value

        self.__rfd_dist_coeff[1,0] = 0.0
        self.__rfd_dist_coeff[2,0] = 0.0
        self.__rfd_dist_coeff[3,0] = 0.0

        self._width, self._height = self.video_capture.size()

        self.__rfd_matrix = np.eye(3, dtype = np.float32)
        self.__rfd_matrix[0,2] = self._width / 2.0
        self.__rfd_matrix[1,2] = self._height / 2.0

        self._mapx, self._mapy = cv2.initUndistortRectifyMap(self.__rfd_matrix, 
            self.__rfd_dist_coeff, np.eye(3), self.__rfd_matrix, 
            (self._width, self._height), cv2.CV_16SC2)

    def read(self):
        # Grab a single frame of video
        ret, frame = self.video_capture.read()

        if ret:
            frame = cv2.remap(frame, self._mapx, self._mapy, cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT)
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

class CLAHEHistogram(ImageFilter):
    def __init__(self, video_capture):
        super().__init__(video_capture)
        self.cl = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16,16))

    def read(self):
        # Grab a single frame of video
        ret, frame = self.video_capture.read()

        if ret:
            img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            # equalize the histogram of the Y channel
            img_yuv[:,:,0] = self.cl.apply(img_yuv[:,:,0])

            # convert the YUV image back to RGB format
            frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            return ret, frame
        else:
            return ret, None

class HSVBrightness(ImageFilter):
    def __init__(self, video_capture):
        super().__init__(video_capture)
        self.cl = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16,16))
        self.value = 1.5

    def read(self):
        # Grab a single frame of video
        ret, frame = self.video_capture.read()

        if ret:
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            current_value = img_hsv[...,2]
            if current_value.all() != 0:
                img_hsv[...,2] = np.where((255/current_value) < self.value, 255, current_value * self.value)

            frame = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            return ret, frame
        else:
            return ret, None

class HSVCLAHEHistogram(ImageFilter):
    def __init__(self, video_capture):
        super().__init__(video_capture)
        self.cl = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))

    def read(self):
        # Grab a single frame of video
        ret, frame = self.video_capture.read()

        if ret:
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            img_hsv[...,2] = self.cl.apply(img_hsv[...,2])

            frame = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

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
        help="parameters to apply histogram")
    ap.add_argument("-cl", "--clahe_histogram", type=bool,
        help="parameters to apply CLAHE histogram")
    ap.add_argument("-hb", "--hsv_brightness", type=bool,
        help="parameters to HSV brightness")
    ap.add_argument("-hch", "--hsv_clahe_histogram", type=bool,
        help="parameters to HSV CLAHE brightness")
    args = vars(ap.parse_args())

    vc = VideoCapture()
    success = vc.set_source(args["video_source"])

    if success:
        if args["fisheye"]:
            vc = FishEye(vc)
            vc.set(value=args["fisheye"])

        if args["histogram"]:
            vc = Histogram(vc)

        if args["clahe_histogram"]:
            vc = CLAHEHistogram(vc)

        if args["hsv_brightness"]:
            vc = HSVBrightness(vc)

        if args["hsv_clahe_histogram"]:
            vc = HSVCLAHEHistogram(vc)

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