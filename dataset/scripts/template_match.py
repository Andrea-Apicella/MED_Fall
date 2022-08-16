import traceback
from collections import OrderedDict

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression
from tqdm import trange
import sys

from utils.utility_functions import listdir_nohidden_sorted


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized




class TemplateMatch:
    """Extraction of timestamps and datalogger time vectors from the NGIMU software video capture using Template Matching algorithm"""

    def __init__(
        self,
        video_path,
        templates_path,
        timestamp_roi=None,
        datalogger_roi=None,
        #(height, width)
        target_template_size=(55, 88) ,
        target_timestamp_size = (800, 200),
        target_datalogger_size =(2444, 260),
    ):
        self.templates_path = templates_path
        self.templates = [cv2.imread(template) for template in listdir_nohidden_sorted(self.templates_path)]
        print(f"Found {len(self.templates)} templates in {self.templates_path}.")

        self.TARGET_TEMPLATE_SIZE = target_template_size
        self.TARGET_TIMESTAMP_SIZE = target_timestamp_size
        self.TARGET_DATALOGGER_SIZE = target_datalogger_size
        self.video_path = video_path
        self.timestamp_roi = timestamp_roi
        self.datalogger_roi = datalogger_roi

        def total_frames(video_path):
            cap = cv2.VideoCapture(video_path)
            _, _ = cap.read()
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total number of frames in the video:", n_frames)
            cap.release()
            return n_frames

        self.total_frames = total_frames(self.video_path)



    def pre_process_templates(self) -> None:
        def pre_process(template):
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            #template = image_resize(template, height=self.TARGET_TEMPLATE_SIZE[0])
            #template = imutils.resize(template, height=self.TARGET_TEMPLATE_SIZE[1])
            template = cv2.resize(template, self.TARGET_TEMPLATE_SIZE)
            template = cv2.bitwise_not(template)
            return template.astype(np.uint8)

        self.templates = list(map(lambda template: pre_process(template), self.templates))

    def __select_roi(self, target):
        cap = cv2.VideoCapture(self.video_path)
        _, first_frame = cap.read()
        roi = cv2.selectROI(str(target), first_frame)
        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        for i in range (1,5):
            cv2.waitKey(1)
        return roi

    def find_rois(self, frame):
        img = frame[:, 0:960]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        d = {}
        # areas = []
        # bounding_rects = []
        for cnt in contours:

            area = cv2.contourArea(cnt)
            bounding_rect = cv2.boundingRect(cnt)
            d[area] = bounding_rect

        d_sorted = sorted(d, reverse=True)
        areas = d_sorted[1:3]
        # rois = []
        # for i in range(len(areas)):
        #     key = areas[i]
        #     print(key)
        #     # (x,y,w,h) = d[key]
        #     print('ROI', d[key])
        #     rois.append(d[key])
            #cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(0,255), randrange(0,255), randrange(0,255)), 2)
        print(d[areas[0]], d[areas[1]])
        return (d[areas[0]], d[areas[1]])

    def template_match(self, frame, element_type, threshold=0.65, test=False):
        '''Extracts timestamp and datalogger strings from current frame'''

        def pre_process(frame):
            frame = cv2.bitwise_not(frame)
            newsize = self.TARGET_TIMESTAMP_SIZE if element_type == "timestamp" else None
            print('newsize', newsize)
            if newsize is not None:
                frame = cv2.resize(frame, dsize=newsize, interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame.astype(np.uint8)

        def show(frame, tomatch, templates):
            """Shows templates, frame and frame ROI from which to extract the digits."""
            _, ax = plt.subplots(nrows=1, ncols=len(templates), figsize=(20, 30))
            for count, col in enumerate(ax):  # type: ignore
                col.imshow(self.templates[count], cmap="gray")
                col.set_title(f"template {count}")
            plt.title("Templates")
            plt.show()

            plt.figure(figsize=(20, 20))
            plt.title("Frame")
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()

            plt.figure()
            plt.title("ROI to perform Template Matching on")
            plt.imshow(tomatch)
            plt.show()

        if len(self.templates[0].shape) > 2:
            self.pre_process_templates()
            print(self.templates[0].shape)
        top_x, top_y, bottom_x, bottom_y = self.timestamp_roi if element_type == "timestamp" else self.datalogger_roi
        if element_type == "timestamp":
            tomatch = frame[top_y+30 : top_y + bottom_y-30, top_x : top_x + bottom_x]
        else:
            tomatch = frame[top_y : top_y + bottom_y, top_x : top_x + bottom_x]
        tomatch = pre_process(tomatch)

        if test:
            show(frame, tomatch, self.templates)
            # sys.exit()

        tomatch_copy = tomatch.copy()

        # dict will contain starting x coordinates of template's digits bounding boxes
        x1Coords = {}

        for template_number, _ in enumerate(self.templates):
            curr_template = self.templates[template_number]
            (tH, tW) = curr_template.shape[:2]
            result = cv2.matchTemplate(tomatch_copy, curr_template, cv2.TM_CCOEFF_NORMED)
            (yCoords, xCoords) = np.where(result >= threshold)

            boxes = []
            for (x, y) in zip(xCoords, yCoords):
                boxes.append((x, y, x + tW, y + tH))
            nms_boxes = non_max_suppression(np.array(boxes))

            if len(nms_boxes) > 1:
                keys = [box[0] for box in nms_boxes]
                for key in keys:
                    x1Coords[key] = template_number
            elif len(nms_boxes) == 1:
                key = nms_boxes[0][0]
                x1Coords[key] = template_number
            else:
                pass

            if test:
                for (startX, startY, endX, endY) in nms_boxes:
                    cv2.rectangle(tomatch_copy, (startX, startY), (endX, endY), (0, 0, 0), 3)
                if template_number == len(self.templates) - 1:
                    plt.figure()
                    plt.imshow(tomatch_copy, cmap="gray")
                    plt.title("Bounding boxes")
                    plt.show()

        x1Coords_unique = OrderedDict(sorted(x1Coords.items()))
        digits = list(x1Coords_unique.values())
        try:
            extracted_format = list("XX:XX:XX.XXX")
            digits_index = 0
            extracted = extracted_format
            for _, char in enumerate(extracted_format):
                if char not in ([":", "."]):
                    extracted[_] = str(digits[digits_index])
                    digits_index += 1
            extracted = "".join(extracted)
            return extracted

        except Exception:
            traceback.print_exc()
            print(f"[ERROR] starts is empty! Zero bounding boxes drawn for this {element_type}.")
            return



    def extract_timestamps_dataloggers(self):
        # if not self.timestamp_roi:
        #     self.timestamp_roi = self.__select_roi("Select Timestamp ROI")
        # if not self.datalogger_roi:
        #     self.datalogger_roi = self.__select_roi("Select Datalogger ROI")
        # print(f"Timestamp ROI: {self.timestamp_roi}")
        # print(f"Datalogger ROI: {self.datalogger_roi}")

        cap = cv2.VideoCapture(self.video_path)
        _, first_frame = cap.read()
        cap.release()


        self.timestamp_roi, self.datalogger_roi = self.find_rois(first_frame)

        # timestamp_test, datalogger_test = self.test_single_frame()
        #
        # if not timestamp_test:
        #     print("Can't extract TIMESTAMP from first frame.")
        #     return
        # if not datalogger_test:
        #     print("Can't extract DATALOGGER from first frame.")
        #     return


        timestamps = []
        dataloggers = []
        cap = cv2.VideoCapture(self.video_path)
        for frame_number in (t := trange(self.total_frames)):
            # for frame_number in trange(self.total_frames):
            success, frame = cap.read()
            if not success:
                print(f"Couldn't read frame {frame_number}")
                break
            timestamp_extracted = self.template_match(frame, element_type="timestamp", test=False)
            datalogger_extracted = self.template_match(frame, element_type="datalogger", test=False)
            timestamps.append(timestamp_extracted)
            dataloggers.append(datalogger_extracted)
            t.set_description(f"Extracted timestamp: {timestamp_extracted}, Extracted datalogger {datalogger_extracted}")
        cap.release()
        return timestamps, dataloggers

    def test_single_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        _, first_frame = cap.read()
        cap.release()

        timestamp_extracted = self.template_match(first_frame, element_type="timestamp")
        datalogger_extracted = self.template_match(first_frame, element_type="datalogger")
        return timestamp_extracted, datalogger_extracted


# def main():
#     pass
#
#
# if __name__ == "__main__":
#     main()
