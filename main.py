from ultralytics import YOLO
import cv2
import argparse
import supervision as sv
import numpy as np

class YOLOv8Live:
    ZONE_POLYGON_LEFT = np.array([
        [0, 0],
        [0.50, 0],
        [0.50, 1],
        [0, 1]
    ])
    
    ZONE_POLYGON_RIGHT = np.array([
        [0.50, 0],
        [1, 0],
        [1, 1],
        [0.50, 1]
    ])
    
    def __init__(self, webcam_resolution):
        self.frame_width, self.frame_height = webcam_resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        self.model = YOLO("yolov8n.pt")
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
        
        self.zone_polygon_left = (self.ZONE_POLYGON_LEFT * np.array(webcam_resolution)).astype(int)
        self.zone_left = sv.PolygonZone(polygon=self.zone_polygon_left, frame_resolution_wh=tuple(webcam_resolution))
        self.zone_annotator_left = sv.PolygonZoneAnnotator(zone=self.zone_left, color=sv.Color.red(), thickness=2, text_thickness=4, text_scale=2)
        
        self.zone_polygon_right = (self.ZONE_POLYGON_RIGHT * np.array(webcam_resolution)).astype(int)
        self.zone_right = sv.PolygonZone(polygon=self.zone_polygon_right, frame_resolution_wh=tuple(webcam_resolution))
        self.zone_annotator_right = sv.PolygonZoneAnnotator(zone=self.zone_right, color=sv.Color.blue(), thickness=2, text_thickness=4, text_scale=2)
        
        self.detections = None  

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="YOLOv8 live")
        parser.add_argument("--webcam-resolution", default=[1200, 720], nargs=2, type=int)
        args = parser.parse_args()
        return args

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            result = self.model(frame)[0]
            self.detections = sv.Detections.from_ultralytics(result)
            self.detections = self.detections[self.detections.class_id != 0]

            labels = [
                f"{self.model.names[class_id]} :: {conf:0.2f}"
                for xyxy, mask, conf, class_id, tracker_id, data in self.detections
            ]

            frame = self.bounding_box_annotator.annotate(scene=frame, detections=self.detections)
            frame = self.label_annotator.annotate(scene=frame, detections=self.detections, labels=labels)

            self.zone_left.trigger(detections=self.detections)
            frame = self.zone_annotator_left.annotate(scene=frame)

            self.zone_right.trigger(detections=self.detections)
            frame = self.zone_annotator_right.annotate(scene=frame)

            yolo_live.left_detections()
            yolo_live.right_detections()

            cv2.imshow("yolov8", frame)
            
            if cv2.waitKey(30) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def left_detections(self):
        if self.detections is not None:
            # print("Checking left detections...")
            for xyxy, mask, conf, class_id, tracker_id, data in self.detections:
                x1, y1, x2, y2 = xyxy
                tlx, tly = self.zone_polygon_left[0]
                brx, bry = self.zone_polygon_left[2]
                
                # print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                # print(f"tlx={tlx}, tly={tly}, brx={brx}, bry={bry}")
                
                if x1 >= tlx and y1 >= tly and x2 <= brx and y2 <= bry:
                    return(f"LEFT Detected class: {self.model.names[class_id]}, Confidence: {conf}")

    def right_detections(self):
        if self.detections is not None:
            # print("Checking right detections...")
            for xyxy, mask, conf, class_id, tracker_id, data in self.detections:
                x1, y1, x2, y2 = xyxy
                tlx, tly = self.zone_polygon_right[0]
                brx, bry = self.zone_polygon_right[2]
                
                # print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                # print(f"tlx={tlx}, tly={tly}, brx={brx}, bry={bry}")
                
                if x1 >= tlx and y1 >= tly and x2 <= brx and y2 <= bry:
                    return(f"RIGHT Detected class: {self.model.names[class_id]}, Confidence: {conf}")

if __name__ == "__main__":
    args = YOLOv8Live.parse_arguments()
    yolo_live = YOLOv8Live(webcam_resolution=args.webcam_resolution)
    yolo_live.run()
    
   
    
