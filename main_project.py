#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
import time
import astar
import main_ui
import argparse
import os
import imutils
import time

lat_in_min = -33.799787
lat_in_max = -33.800252
lat_out_min = 0
lat_out_max = 720

long_in_min = 151.156630
long_in_max = 151.157704
long_out_min = 0
long_out_max = 1280

map_image = cv2.imread("map_data/map.jpg")
map_image = cv2.resize(map_image, (1280, 720))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="flight_recordings/DJI_0048.MP4", help="path to input video")
ap.add_argument("-y", "--yolo", default="yolo-drone", help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.7, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "drone.names"])
LABELS = open(labelsPath).read().strip().split("\n")

coordinates_file = open('flight_data/coordinates_final.txt')

np.random.seed(42)

weightsPath = os.path.sep.join([args["yolo"], "yolov4-drone_best.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4-drone.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(args["input"])

class VideoThread(QtCore.QThread):
	drone_feed_signal = QtCore.pyqtSignal(np.ndarray)
	drone_position_to_map = QtCore.pyqtSignal(tuple)

	latitude_value = QtCore.pyqtSignal(str)
	longitude_value = QtCore.pyqtSignal(str)
	status_string = QtCore.pyqtSignal(str)

	person_found_value = QtCore.pyqtSignal(str)
	hazards_found_value = QtCore.pyqtSignal(str)

	def map_lat(self, input):
	    return lat_out_min + ((lat_out_max - lat_out_min) / (lat_in_max - lat_in_min)) * (input - lat_in_min)

	def map_long(self, input):
	    return long_out_min + ((long_out_max - long_out_min) / (long_in_max - long_in_min)) * (input - long_in_min)

	def run(self):
		(W, H) = (None, None)
		j=0
		while True:
			j+=1

			if j>=5:
				for i in range (0,1):
					coordinates_file.readline()
				real = coordinates_file.readline()
				latitude = real.split("\t")[0]
				longitude = real.split("\t")[1]
				status = real.split("\t")[2]
				trigger = real.split("\t")[3]
				self.latitude_value.emit(latitude)
				self.longitude_value.emit(longitude)
				self.status_string.emit(status)

				if "person" in trigger:
					person_found_count = "1"
					self.person_found_value.emit(person_found_count)
				elif "fire" in trigger:
					hazards_found_count = "1"
					self.hazards_found_value.emit(hazards_found_count)

				j=0
				y_pos = int(self.map_lat(float(latitude)))
				x_pos = int(self.map_long(float(longitude)))
				self.drone_position_to_map.emit((x_pos, y_pos, trigger))

			(grabbed, frame) = vs.read()
			if not grabbed:
				print("Breaking.")
				break
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			end = time.time()
			elap = end - start

			boxes = []
			confidences = []
			classIDs = []

			for output in layerOutputs:
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					if confidence > args["confidence"]:
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

			if len(idxs) > 0:
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					centerX = x + h/2
					centerY = y + w/2

					if LABELS[classIDs[i]] == "person":
						colour = (255,255,0)
					elif LABELS[classIDs[i]] == "fire":
						colour = (0,0,255)
					cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
			self.drone_feed_signal.emit(frame)
		print("[INFO] cleaning up...")
		vs.release()
		coordinates_file.close()
		print("[INFO] vs released...")

class PathThread(QtCore.QThread):
	def __init__(self):
		super().__init__()
		self.ready = False
		self.start_x = 0
		self.start_y = 0
		self.goal_x = 0
		self.goal_y = 0
		self.hazard = None

	path_map = QtCore.pyqtSignal(np.ndarray)

	def receive_drone_position(self, data):
		cv2.circle(map_image, (data[0], data[1]), 3, (255,0,242), -1)
		if "person" in data[2]:
			cv2.circle(map_image, (data[0], data[1]), 30, (255,132,0), 3)
			self.goal_x = data[0]
			self.goal_y = data[1]
		elif "fire" in data[2]:
			cv2.circle(map_image, (data[0], data[1]), 70, (0,0,255), 3)
			self.hazard = (data[0], data[1])
		elif "home" in data[2]:
			self.start_x = data[0]
			self.start_y = data[1]
		self.path_map.emit(map_image)

	def readynow(self):
		self.ready = True

	def run(self):
		while True:
			while not self.ready:
				time.sleep(1)
			result, path_to_take = astar.main(self.start_x, self.start_y, self.goal_x, self.goal_y, self.hazard)
			for p in path_to_take:
				cv2.circle(map_image, (p[0], p[1]), 4, (0,255,0), -1)
			self.path_map.emit(map_image)
			self.ready = False

class main_window(main_ui.Ui_MainWindow):
	def setupUi(self, MainWindow):
		super().setupUi(MainWindow)

		self.video_thread = VideoThread()
		self.path_thread = PathThread()

		self.video_thread.drone_feed_signal.connect(self.update_drone_feed)
		self.video_thread.latitude_value.connect(self.set_latitude)
		self.video_thread.longitude_value.connect(self.set_longitude)
		self.video_thread.status_string.connect(self.set_status_text)
		self.video_thread.person_found_value.connect(self.set_persons_found)
		self.video_thread.hazards_found_value.connect(self.set_hazards_found)
		self.video_thread.drone_position_to_map.connect(self.path_thread.receive_drone_position)

		self.path_thread.path_map.connect(self.update_map)

		self.video_thread.start()
		self.path_thread.start()

		self.plot_path_button.clicked.connect(self.path_thread.readynow)

	def set_latitude(self, value):
		self.latitude_value.setProperty("value", value)

	def set_longitude(self, value):
		self.longitude_value.setProperty("value", value)

	def set_persons_found(self, value):
		palette = QtGui.QPalette()
		brush = QtGui.QBrush(QtGui.QColor(138, 226, 52))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
		brush = QtGui.QBrush(QtGui.QColor(198, 255, 143))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
		brush = QtGui.QBrush(QtGui.QColor(168, 240, 97))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
		brush = QtGui.QBrush(QtGui.QColor(69, 113, 26))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
		brush = QtGui.QBrush(QtGui.QColor(92, 151, 34))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
		brush = QtGui.QBrush(QtGui.QColor(138, 226, 52))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
		brush = QtGui.QBrush(QtGui.QColor(138, 226, 52))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
		self.person_found_number.setPalette(palette)
		self.person_found_number.setProperty("value", value)

	def set_hazards_found(self, value):
		palette = QtGui.QPalette()
		brush = QtGui.QBrush(QtGui.QColor(239, 41, 41))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
		brush = QtGui.QBrush(QtGui.QColor(255, 147, 147))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
		brush = QtGui.QBrush(QtGui.QColor(247, 94, 94))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
		brush = QtGui.QBrush(QtGui.QColor(119, 20, 20))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
		brush = QtGui.QBrush(QtGui.QColor(159, 27, 27))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
		brush = QtGui.QBrush(QtGui.QColor(239, 41, 41))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
		brush = QtGui.QBrush(QtGui.QColor(239, 41, 41))
		brush.setStyle(QtCore.Qt.SolidPattern)
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
		self.hazards_found_number.setPalette(palette)
		self.hazards_found_number.setProperty("value", value)

	def set_status_text(self, text):
		self.status_value.setText(text)

	def update_drone_feed(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.drone_feed_display.setPixmap(qt_img)

	def update_map(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.map_display.setPixmap(qt_img)

	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		p = convert_to_Qt_format.scaled(1920, 1080, QtCore.Qt.KeepAspectRatio)
		return QtGui.QPixmap.fromImage(p)


if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = main_window()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())