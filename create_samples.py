"""
Generates high-quality clean sample traffic sign images for instant testing.
"""
import os
import cv2
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

def make_stop_sign():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Red Octagon
    pts = np.array([[120, 40], [280, 40], [360, 120], [360, 280],
                    [280, 360], [120, 360], [40, 280], [40, 120]], np.int32)
    cv2.fillPoly(img, [pts], (25, 25, 215)) # Red in BGR
    cv2.polylines(img, [pts], True, (255, 255, 255), 10)
    # Text STOP
    cv2.putText(img, "STOP", (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 255), 10)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_stop.jpg"), img)

def make_speed_limit_50():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Red outer ring
    cv2.circle(img, (200, 200), 160, (25, 25, 215), -1)
    # White inner circle
    cv2.circle(img, (200, 200), 125, (255, 255, 255), -1)
    # Text 50
    cv2.putText(img, "50", (95, 245), cv2.FONT_HERSHEY_SIMPLEX, 3.8, (20, 20, 20), 12)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_speed_50.jpg"), img)

def make_speed_limit_80():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Red outer ring
    cv2.circle(img, (200, 200), 160, (25, 25, 215), -1)
    # White inner circle
    cv2.circle(img, (200, 200), 125, (255, 255, 255), -1)
    # Text 80
    cv2.putText(img, "80", (95, 245), cv2.FONT_HERSHEY_SIMPLEX, 3.8, (20, 20, 20), 12)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_speed_80.jpg"), img)

def make_yield_sign():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Red Inverted Triangle
    pts_out = np.array([[50, 60], [350, 60], [200, 350]], np.int32)
    cv2.fillPoly(img, [pts_out], (25, 25, 215))
    pts_in = np.array([[95, 85], [305, 85], [200, 305]], np.int32)
    cv2.fillPoly(img, [pts_in], (255, 255, 255))
    cv2.putText(img, "YIELD", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (25, 25, 215), 4)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_yield.jpg"), img)

def make_pedestrian():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Blue Square
    cv2.rectangle(img, (50, 50), (350, 350), (210, 110, 20), -1) # Blue in BGR
    # White inside triangle
    pts = np.array([[200, 70], [70, 330], [330, 330]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    # Person symbol in black
    cv2.circle(img, (200, 140), 20, (20, 20, 20), -1)
    cv2.line(img, (200, 160), (200, 240), (20, 20, 20), 10)
    cv2.line(img, (200, 180), (160, 220), (20, 20, 20), 8)
    cv2.line(img, (200, 180), (240, 210), (20, 20, 20), 8)
    cv2.line(img, (200, 240), (170, 300), (20, 20, 20), 10)
    cv2.line(img, (200, 240), (230, 300), (20, 20, 20), 10)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_pedestrian.jpg"), img)

def make_turn_right():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Blue Circle
    cv2.circle(img, (200, 200), 160, (210, 110, 20), -1)
    cv2.circle(img, (200, 200), 154, (255, 255, 255), 4)
    # Right arrow
    arrow_pts = np.array([[120, 220], [120, 180], [220, 180], [220, 130], [300, 200], [220, 270], [220, 220]], np.int32)
    cv2.fillPoly(img, [arrow_pts], (255, 255, 255))
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_turn_right.jpg"), img)

def make_no_entry():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Red Solid Circle
    cv2.circle(img, (200, 200), 160, (25, 25, 215), -1)
    # White horizontal bar
    cv2.rectangle(img, (80, 175), (320, 225), (255, 255, 255), -1)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_no_entry.jpg"), img)

def make_traffic_light():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Yellow triangle
    pts = np.array([[200, 40], [370, 350], [30, 350]], np.int32)
    cv2.fillPoly(img, [pts], (20, 200, 240)) # Yellow
    cv2.polylines(img, [pts], True, (25, 25, 215), 14)
    # Traffic signal box
    cv2.rectangle(img, (170, 140), (230, 290), (30, 30, 30), -1)
    cv2.circle(img, (200, 165), 16, (25, 25, 220), -1) # Red light
    cv2.circle(img, (200, 215), 16, (20, 200, 240), -1) # Yellow light
    cv2.circle(img, (200, 265), 16, (40, 210, 40), -1) # Green light
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_traffic_light.jpg"), img)

def make_school_zone():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Yellow Pentagram / Triangle
    pts = np.array([[200, 40], [370, 350], [30, 350]], np.int32)
    cv2.fillPoly(img, [pts], (20, 200, 240))
    cv2.polylines(img, [pts], True, (25, 25, 215), 14)
    cv2.putText(img, "SCHOOL", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 4)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_school_zone.jpg"), img)

def make_road_work():
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    # Yellow diamond
    pts = np.array([[200, 40], [360, 200], [200, 360], [40, 200]], np.int32)
    cv2.fillPoly(img, [pts], (20, 170, 245)) # Orange/Yellow
    cv2.polylines(img, [pts], True, (20, 20, 20), 10)
    cv2.putText(img, "WORK", (125, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 5)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_road_work.jpg"), img)

if __name__ == "__main__":
    make_stop_sign()
    make_speed_limit_50()
    make_speed_limit_80()
    make_yield_sign()
    make_pedestrian()
    make_turn_right()
    make_no_entry()
    make_traffic_light()
    make_school_zone()
    make_road_work()
    print("Generated 10 clean sample traffic sign images in static/samples!")
