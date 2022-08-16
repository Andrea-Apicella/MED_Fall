import argparse

import cv2
import pytesseract
import numpy as np
import re


def total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    _, _ = cap.read()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames in the video:", n_frames)
    cap.release()
    return n_frames


def find_rois(frame) -> tuple[np.array, np.array]:
    img = frame[:, 0:960]
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


def enhance_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

video_path = '/Users/andrea/Documents/Github/MED_Fall/dataset/data/Actor_1_Bed_Rolling CAM 8.mp4'

cap = cv2.VideoCapture(video_path)

n_frames = total_frames(video_path)

l = []

filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
for n in range(n_frames):
    ret, frame = cap.read()
    timestamp_image_region, datalogger_image_region = find_rois(frame)

    timestamp_img = enhance_text(timestamp_image_region)
    datalogger_img = enhance_text(datalogger_image_region)


    timestamp_text = pytesseract.image_to_string(timestamp_img)
    datalogger_text = pytesseract.image_to_string(datalogger_img)

    timestamp_text = re.sub('\D', '', timestamp_text)
    datalogger_text = re.sub('\D', '', datalogger_text)
    if len(timestamp_text) != 9:
        print("timestamp_text wrong", n)
        cv2.imshow("faulty timestamp", timestamp_img)
        cv2.waitKey(0)
    if len(datalogger_text) != 9:
        print("datalogger_text wrong", n)
        cv2.imshow("faulty datalogger", datalogger_img)
        cv2.waitKey(0)

    l.append("timestamp_text: {timestamp_text}, datalogger_text: {datalogger_text}")
    print(f"timestamp_text: {timestamp_text}, datalogger_text: {datalogger_text}")
print(len(l))


