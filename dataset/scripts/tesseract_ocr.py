import argparse

import cv2
import pytesseract
import numpy as np
import re
from tqdm.auto import trange


class TesseractOCR:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def __total_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        _, _ = cap.read()
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total number of frames in the video:", n_frames)
        cap.release()
        return n_frames

    def __enhance_text(self, image: np.array) -> np.array:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh

    def __strip_unwanted_chars_validate_format(self, text: str, n: int):

        text = re.sub('\D', '', text)
        assert len(text) == 9, print(f"Faulty extraction at frame {n}. Text extracted: {text}")

        text = f"{text[0:2]}:{text[2:4]}:{text[4:6]}.{text[6:]}"
        return text

    def find_rois(self, frame: np.array) -> tuple[np.array, np.array]:

        h, w = frame.shape[:-1]
        img = frame[:, 0:int(w / 2)]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        d = {}

        for cnt in contours:
            area = cv2.contourArea(cnt)
            bounding_rect = cv2.boundingRect(cnt)
            d[area] = bounding_rect

        d_sorted = sorted(d, reverse=True)
        areas = d_sorted[1:3]

        timestamp_roi = d[areas[0]]
        datalogger_roi = d[areas[1]]

        top_x, top_y, bottom_x, bottom_y = timestamp_roi
        timestamp_image_region = frame[top_y: top_y + bottom_y, top_x: top_x + bottom_x]
        top_x, top_y, bottom_x, bottom_y = datalogger_roi
        datalogger_image_region = frame[top_y: top_y + bottom_y, top_x: top_x + bottom_x]

        return timestamp_image_region, datalogger_image_region

    def extract_timestamp_datalogger(self) -> tuple[list, list]:

        n_frames = self.__total_frames()
        cap = cv2.VideoCapture(self.video_path)

        timestamps = [None] * n_frames
        dataloggers = [None] * n_frames

        for n in (t := trange(n_frames)):
            ret, frame = cap.read()
            timestamp_image_region, datalogger_image_region = self.find_rois(frame)

            timestamp_img = self.__enhance_text(timestamp_image_region)
            datalogger_img = self.__enhance_text(datalogger_image_region)

            timestamp_text = pytesseract.image_to_string(timestamp_img)
            datalogger_text = pytesseract.image_to_string(datalogger_img)

            timestamp_text = re.sub('\D', '', timestamp_text)
            datalogger_text = re.sub('\D', '', datalogger_text)

            timestamps[n] = self.__strip_unwanted_chars_validate_format(timestamp_text, n)
            dataloggers[n] = self.__strip_unwanted_chars_validate_format(datalogger_text, n)

            t.set_description(f"Frame {n}. Timestamp: {timestamps[n]}, Datalogger: {dataloggers[n]}")
        return timestamps, dataloggers
