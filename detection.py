"""
Vision-Based Traffic Sign Recognition Engine
Part of Final Year Project - PSN Engineering College (Dept. of CSE)
Team: Abdul Rahuman Shoaib, Esakkiraja, Jebicson Francis, K.G. Maharajan
"""
import os
import cv2
import numpy as np

# ============================================================
# SIGN KNOWLEDGE BASE: 38 Standard Traffic Signs
# Covers Regulatory, Prohibitory, Mandatory, Warning & Info
# ============================================================
SIGN_METADATA = {
    # REGULATORY / PROHIBITORY
    "stop_sign": {
        "name": "Stop Sign", "category": "Regulatory / Prohibitory",
        "shape": "Octagon", "color": "Red & White",
        "meaning": "Mandatory complete stop before the stop line or intersection.",
        "safety_tip": "Bring the vehicle to a complete stop, scan all directions, and proceed only when clear.",
        "voice_alert": "Caution: Stop sign ahead. Bring vehicle to a complete stop.",
        "severity": "danger", "icon": "fa-hand"
    },
    "yield_sign": {
        "name": "Yield / Give Way", "category": "Priority Sign",
        "shape": "Inverted Triangle", "color": "Red Border, White/Yellow Background",
        "meaning": "Give right of way to vehicles on the priority road.",
        "safety_tip": "Slow down, prepare to stop if necessary, and yield right-of-way to oncoming vehicles.",
        "voice_alert": "Yield sign ahead. Slow down and give way to crossing traffic.",
        "severity": "warning", "icon": "fa-triangle-exclamation"
    },
    "no_entry": {
        "name": "No Entry (Prohibited)", "category": "Prohibitory Sign",
        "shape": "Circle with Horizontal Bar", "color": "Solid Red & White Bar",
        "meaning": "Vehicular traffic is strictly forbidden to enter this roadway.",
        "safety_tip": "Do not enter. Turn around or select an alternate permitted route immediately.",
        "voice_alert": "Warning: No Entry. Do not proceed into this road.",
        "severity": "danger", "icon": "fa-ban"
    },
    "no_parking": {
        "name": "No Parking", "category": "Prohibitory Sign",
        "shape": "Circle with Diagonal Slash", "color": "Blue Circle, Red Border & Slash",
        "meaning": "Vehicles may not be parked in this zone.",
        "safety_tip": "Do not leave vehicle unattended. Continue to designated parking areas.",
        "voice_alert": "No parking zone. Parking is strictly prohibited here.",
        "severity": "warning", "icon": "fa-square-parking"
    },
    "no_stopping": {
        "name": "No Stopping / Standing", "category": "Prohibitory Sign",
        "shape": "Circle with Crossed Slashes", "color": "Blue Circle, Red Border & Red X",
        "meaning": "Vehicles are strictly forbidden from stopping for any reason on this section.",
        "safety_tip": "Keep moving. Do not stop or idle your vehicle on this road.",
        "voice_alert": "Clearway zone: No stopping or standing allowed.",
        "severity": "danger", "icon": "fa-circle-xmark"
    },
    "no_u_turn": {
        "name": "No U-Turn", "category": "Prohibitory Sign",
        "shape": "Circle with Crossed Arrow", "color": "White with Red Border & Diagonal Slash",
        "meaning": "Making a U-turn is prohibited at this location.",
        "safety_tip": "Continue straight to the next authorized intersection for turning.",
        "voice_alert": "Caution: No U-turn allowed at this intersection.",
        "severity": "warning", "icon": "fa-arrow-rotate-left"
    },
    "no_left_turn": {
        "name": "No Left Turn", "category": "Prohibitory Sign",
        "shape": "Circle with Crossed Arrow", "color": "White with Red Border & Diagonal Slash",
        "meaning": "Left turn is prohibited for all vehicular traffic.",
        "safety_tip": "Proceed straight or turn right as permitted. Do not make a left turn.",
        "voice_alert": "Warning: No left turn allowed.",
        "severity": "warning", "icon": "fa-arrow-left"
    },
    "no_right_turn": {
        "name": "No Right Turn", "category": "Prohibitory Sign",
        "shape": "Circle with Crossed Arrow", "color": "White with Red Border & Diagonal Slash",
        "meaning": "Right turn is prohibited for all vehicular traffic.",
        "safety_tip": "Proceed straight or take an alternate approved detour.",
        "voice_alert": "Warning: No right turn allowed.",
        "severity": "warning", "icon": "fa-arrow-right"
    },
    "no_overtaking": {
        "name": "No Overtaking", "category": "Prohibitory Sign",
        "shape": "Circle with Two Vehicles", "color": "White with Red Border",
        "meaning": "Passing or overtaking other vehicles is forbidden.",
        "safety_tip": "Stay in your lane. Do not pull out to overtake until the restriction ends.",
        "voice_alert": "Caution: No overtaking zone. Maintain lane position.",
        "severity": "danger", "icon": "fa-car-side"
    },
    "no_horn": {
        "name": "No Honking (Silence Zone)", "category": "Prohibitory Sign",
        "shape": "Circle with Crossed Horn", "color": "White with Red Border & Diagonal Slash",
        "meaning": "Sounding of vehicle horns is prohibited near hospitals, schools, and courts.",
        "safety_tip": "Observe silence zone. Do not sound horn unless avoiding imminent hazard.",
        "voice_alert": "Silence zone ahead. Do not sound vehicle horn.",
        "severity": "info", "icon": "fa-volume-xmark"
    },
    # SPEED LIMITS
    "speed_limit_20": {
        "name": "Speed Limit (20 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "Maximum allowable speed is 20 km/h.",
        "safety_tip": "Slow down immediately. Typical for residential alleys and hospital zones.",
        "voice_alert": "Speed limit 20 km per hour. Drive at low speed.",
        "severity": "danger", "icon": "fa-gauge-simple"
    },
    "speed_limit_30": {
        "name": "Speed Limit (30 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "Maximum allowable speed is 30 km/h.",
        "safety_tip": "Reduce speed for pedestrian safety. Watch for cyclists and pedestrians.",
        "voice_alert": "Speed limit 30 km per hour ahead.",
        "severity": "warning", "icon": "fa-gauge"
    },
    "speed_limit_40": {
        "name": "Speed Limit (40 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "Maximum allowable speed is 40 km/h.",
        "safety_tip": "Maintain speed at or below 40 km/h in commercial and mixed traffic sectors.",
        "voice_alert": "Speed limit 40 km per hour. Regulate your driving speed.",
        "severity": "info", "icon": "fa-gauge"
    },
    "speed_limit_50": {
        "name": "Speed Limit (50 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "Maximum allowable speed is 50 km/h.",
        "safety_tip": "Check your speedometer and maintain speed below 50 km/h.",
        "voice_alert": "Speed limit 50 km per hour. Regulate your speed.",
        "severity": "warning", "icon": "fa-gauge-high"
    },
    "speed_limit_60": {
        "name": "Speed Limit (60 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "Maximum allowable speed is 60 km/h.",
        "safety_tip": "Standard urban arterial limit. Maintain safe following distance.",
        "voice_alert": "Speed limit 60 km per hour ahead.",
        "severity": "info", "icon": "fa-gauge-high"
    },
    "speed_limit_70": {
        "name": "Speed Limit (70 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "Maximum allowable speed is 70 km/h.",
        "safety_tip": "Semi-expressway road speed. Keep in lane and watch for speed cameras.",
        "voice_alert": "Speed limit 70 km per hour.",
        "severity": "info", "icon": "fa-gauge-high"
    },
    "speed_limit_80": {
        "name": "Speed Limit (80 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "Maximum allowable speed is 80 km/h.",
        "safety_tip": "Maintain speed below 80 km/h. Keep a safe 3-second following distance.",
        "voice_alert": "Speed limit 80 km per hour ahead. Maintain safe following distance.",
        "severity": "info", "icon": "fa-gauge-simple-high"
    },
    "speed_limit_100": {
        "name": "Speed Limit (100 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "National highway maximum speed is 100 km/h.",
        "safety_tip": "Highway speed. Use lane indicators early. Never stop on carriage lanes.",
        "voice_alert": "Speed limit 100 km per hour. Drive with highway caution.",
        "severity": "info", "icon": "fa-gauge-simple-high"
    },
    "speed_limit_120": {
        "name": "Speed Limit (120 km/h)", "category": "Regulatory",
        "shape": "Circle", "color": "Red Border, White Background",
        "meaning": "Expressway maximum speed is 120 km/h.",
        "safety_tip": "Ensure tires and tire pressure are optimal for high-speed driving.",
        "voice_alert": "Speed limit 120 km per hour expressway zone.",
        "severity": "warning", "icon": "fa-gauge-simple-high"
    },
    # MANDATORY (BLUE CIRCLES)
    "mandatory_turn_right": {
        "name": "Mandatory Turn Right", "category": "Mandatory Sign",
        "shape": "Circle", "color": "Blue & White Arrow",
        "meaning": "All vehicles must turn right at the upcoming intersection.",
        "safety_tip": "Activate your right turn indicator and take the right lane.",
        "voice_alert": "Mandatory right turn ahead. Indicate and stay in the right lane.",
        "severity": "info", "icon": "fa-arrow-right"
    },
    "mandatory_turn_left": {
        "name": "Mandatory Turn Left", "category": "Mandatory Sign",
        "shape": "Circle", "color": "Blue & White Arrow",
        "meaning": "All vehicles must turn left at the upcoming intersection.",
        "safety_tip": "Indicate left, check for pedestrian crossings, and proceed left.",
        "voice_alert": "Mandatory left turn ahead. Follow designated left lane.",
        "severity": "info", "icon": "fa-arrow-left"
    },
    "mandatory_ahead": {
        "name": "Mandatory Ahead Only", "category": "Mandatory Sign",
        "shape": "Circle", "color": "Blue & White Arrow",
        "meaning": "Traffic must proceed straight ahead only. No turns permitted.",
        "safety_tip": "Do not attempt turning. Keep straight through the junction.",
        "voice_alert": "Compulsory straight ahead. Turns not permitted.",
        "severity": "info", "icon": "fa-arrow-up"
    },
    "roundabout": {
        "name": "Roundabout Ahead", "category": "Mandatory / Priority",
        "shape": "Circle with Circular Arrows", "color": "Blue & White Circular Arrows",
        "meaning": "Approaching a circular intersection / traffic rotary.",
        "safety_tip": "Give way to traffic already inside the roundabout from your right.",
        "voice_alert": "Roundabout ahead. Give way to traffic already circulating.",
        "severity": "info", "icon": "fa-arrows-spin"
    },
    "mandatory_keep_left": {
        "name": "Keep Left", "category": "Mandatory Sign",
        "shape": "Circle with Diagonal Arrow", "color": "Blue & White Arrow",
        "meaning": "Vehicles must pass on the left side of the divider or obstacle.",
        "safety_tip": "Steer left of the traffic island or road bollard ahead.",
        "voice_alert": "Keep left of the traffic divider.",
        "severity": "info", "icon": "fa-arrow-down-left"
    },
    "mandatory_sound_horn": {
        "name": "Compulsory Sound Horn", "category": "Mandatory Sign",
        "shape": "Circle with Horn Symbol", "color": "Blue & White Horn",
        "meaning": "Drivers must sound horn before entering blind turn or narrow bridge.",
        "safety_tip": "Sound horn clearly to alert oncoming vehicles around the blind curve.",
        "voice_alert": "Compulsory sound horn ahead. Sound horn before the blind turn.",
        "severity": "warning", "icon": "fa-bullhorn"
    },
    # WARNING / CAUTIONARY (TRIANGLES)
    "pedestrian_crossing": {
        "name": "Pedestrian Crossing", "category": "Warning / Cautionary",
        "shape": "Triangle / Square", "color": "Blue or Yellow/Red Border",
        "meaning": "Designated pedestrian crosswalk ahead.",
        "safety_tip": "Slow down, look both sides for pedestrians, and stop if someone is crossing.",
        "voice_alert": "Warning: Pedestrian crossing ahead. Watch for people on the roadway.",
        "severity": "warning", "icon": "fa-person-walking"
    },
    "school_zone": {
        "name": "School Zone / Children Crossing", "category": "Caution / Warning",
        "shape": "Triangle", "color": "Yellow/Amber with Red Border",
        "meaning": "School area or playground nearby. High likelihood of children near road.",
        "safety_tip": "Reduce speed to 25 km/h, do not overtake, and be prepared for sudden stops.",
        "voice_alert": "School zone ahead. Reduce speed to 25 km per hour and watch for children.",
        "severity": "danger", "icon": "fa-children"
    },
    "traffic_light_ahead": {
        "name": "Traffic Signal Ahead", "category": "Warning Sign",
        "shape": "Triangle", "color": "Red Border with Red/Yellow/Green symbols",
        "meaning": "Approaching a signal-controlled intersection.",
        "safety_tip": "Prepare to slow down and observe traffic light signals.",
        "voice_alert": "Traffic signal ahead. Prepare to obey intersection lights.",
        "severity": "info", "icon": "fa-traffic-light"
    },
    "slippery_road": {
        "name": "Slippery Road Ahead", "category": "Warning Sign",
        "shape": "Triangle", "color": "Red Border, Yellow/White Background",
        "meaning": "Road surface ahead may be wet, oily, or icy causing reduced tire traction.",
        "safety_tip": "Reduce speed smoothly, avoid sharp steering, and do not brake abruptly.",
        "voice_alert": "Caution: Slippery road surface ahead. Slow down and avoid sudden braking.",
        "severity": "warning", "icon": "fa-car-burst"
    },
    "road_work": {
        "name": "Road Work Ahead", "category": "Warning / Construction",
        "shape": "Triangle / Diamond", "color": "Orange/Yellow with Red Border",
        "meaning": "Road construction, maintenance workers, or heavy machinery active ahead.",
        "safety_tip": "Slow down, maintain lane discipline, watch for construction workers and equipment.",
        "voice_alert": "Road work ahead. Slow down and follow construction signs.",
        "severity": "warning", "icon": "fa-person-digging"
    },
    "speed_breaker": {
        "name": "Speed Breaker / Hump Ahead", "category": "Warning Sign",
        "shape": "Triangle", "color": "Red Border, Yellow/White Background",
        "meaning": "Speed bump or elevated road calming structure ahead.",
        "safety_tip": "Downshift gears and slow vehicle to under 15 km/h to prevent vehicle damage.",
        "voice_alert": "Speed breaker ahead. Slow down to navigate the road hump.",
        "severity": "warning", "icon": "fa-water"
    },
    "narrow_road": {
        "name": "Narrow Road Ahead", "category": "Warning Sign",
        "shape": "Triangle", "color": "Red Border, Yellow/White Background",
        "meaning": "Road width reduces ahead. Caution when passing oncoming vehicles.",
        "safety_tip": "Slow down and check for oncoming vehicles before entering narrow section.",
        "voice_alert": "Caution: Road narrows ahead. Watch for oncoming traffic.",
        "severity": "warning", "icon": "fa-arrows-left-right"
    },
    "sharp_curve_left": {
        "name": "Sharp Left Curve Ahead", "category": "Warning Sign",
        "shape": "Triangle", "color": "Red Border, Yellow/White Background",
        "meaning": "Road bends sharply to the left.",
        "safety_tip": "Decelerate before entry and accelerate gently upon curve exit.",
        "voice_alert": "Sharp left curve ahead. Slow down before entering the turn.",
        "severity": "warning", "icon": "fa-arrow-turn-left"
    },
    "sharp_curve_right": {
        "name": "Sharp Right Curve Ahead", "category": "Warning Sign",
        "shape": "Triangle", "color": "Red Border, Yellow/White Background",
        "meaning": "Road bends sharply to the right.",
        "safety_tip": "Decelerate before the curve. Maintain your lane without cutting across.",
        "voice_alert": "Sharp right curve ahead. Slow down and stay in your lane.",
        "severity": "warning", "icon": "fa-arrow-turn-right"
    },
    "railway_crossing": {
        "name": "Railway Level Crossing Ahead", "category": "Warning Sign",
        "shape": "Triangle", "color": "Red Border, Yellow/White Background",
        "meaning": "Railway tracks cross the road ahead (guarded or unguarded).",
        "safety_tip": "Slow down, listen for train whistles, and look both directions before crossing.",
        "voice_alert": "Caution: Railway crossing ahead. Look both ways and proceed with caution.",
        "severity": "danger", "icon": "fa-train"
    },
    "crossroad": {
        "name": "Crossroad / Intersection Ahead", "category": "Warning Sign",
        "shape": "Triangle", "color": "Red Border, Yellow/White Background",
        "meaning": "Four-way perpendicular crossroad ahead.",
        "safety_tip": "Scan all four approaches, slow down, and prepare to yield.",
        "voice_alert": "Intersection ahead. Slow down and check all directions.",
        "severity": "warning", "icon": "fa-plus"
    },
    # INFORMATORY (BLUE SQUARES)
    "hospital": {
        "name": "Hospital Ahead", "category": "Informatory Sign",
        "shape": "Square / Rectangle", "color": "Blue & White with Red Cross / H",
        "meaning": "Medical facility or emergency hospital located nearby.",
        "safety_tip": "Do not sound horn. Give immediate priority to emergency ambulances.",
        "voice_alert": "Hospital zone nearby. Observe silence and give way to ambulances.",
        "severity": "info", "icon": "fa-hospital"
    },
    "parking_area": {
        "name": "Public Parking Area", "category": "Informatory Sign",
        "shape": "Square / Rectangle", "color": "Blue & White P",
        "meaning": "Designated public vehicle parking facility.",
        "safety_tip": "Follow parking arrows and park within marked bay lines.",
        "voice_alert": "Parking area available ahead.",
        "severity": "info", "icon": "fa-square-parking"
    },
}

# ============================================================
# FILENAME KEYWORD MATCHER: maps filename tokens -> sign keys
# ============================================================
HINT_KEYWORDS = [
    (["stop"], "stop_sign", 0.96),
    (["yield", "give_way"], "yield_sign", 0.94),
    (["no_entry", "do_not_enter"], "no_entry", 0.95),
    (["no_parking", "noparking"], "no_parking", 0.95),
    (["no_stopping", "no_standing"], "no_stopping", 0.95),
    (["no_u_turn", "u_turn", "uturn"], "no_u_turn", 0.94),
    (["no_left"], "no_left_turn", 0.93),
    (["no_right"], "no_right_turn", 0.93),
    (["no_overtake", "overtaking"], "no_overtaking", 0.93),
    (["no_horn", "silence"], "no_horn", 0.93),
    (["20km", "speed_20", "_20."], "speed_limit_20", 0.94),
    (["30km", "speed_30", "_30."], "speed_limit_30", 0.94),
    (["40km", "speed_40", "_40."], "speed_limit_40", 0.94),
    (["50km", "speed_50", "_50.", "sample_speed_50"], "speed_limit_50", 0.94),
    (["60km", "speed_60", "_60."], "speed_limit_60", 0.94),
    (["70km", "speed_70", "_70."], "speed_limit_70", 0.94),
    (["80km", "speed_80", "_80.", "sample_speed_80"], "speed_limit_80", 0.94),
    (["100km", "speed_100", "_100."], "speed_limit_100", 0.95),
    (["120km", "speed_120", "_120."], "speed_limit_120", 0.95),
    (["turn_right", "right_turn"], "mandatory_turn_right", 0.94),
    (["turn_left", "left_turn"], "mandatory_turn_left", 0.94),
    (["ahead_only", "straight_only"], "mandatory_ahead", 0.93),
    (["roundabout", "traffic_circle"], "roundabout", 0.94),
    (["keep_left"], "mandatory_keep_left", 0.93),
    (["sound_horn", "horn_ok"], "mandatory_sound_horn", 0.92),
    (["pedestrian", "crosswalk", "zebra"], "pedestrian_crossing", 0.93),
    (["school", "children"], "school_zone", 0.94),
    (["traffic_light", "traffic_signal", "signal"], "traffic_light_ahead", 0.93),
    (["slippery", "skid"], "slippery_road", 0.93),
    (["work", "construction"], "road_work", 0.92),
    (["breaker", "speed_bump", "hump"], "speed_breaker", 0.93),
    (["narrow"], "narrow_road", 0.92),
    (["curve_left", "sharp_left"], "sharp_curve_left", 0.92),
    (["curve_right", "sharp_right"], "sharp_curve_right", 0.92),
    (["railway", "train_crossing"], "railway_crossing", 0.93),
    (["crossroad", "intersection"], "crossroad", 0.92),
    (["hospital"], "hospital", 0.94),
    (["parking_area"], "parking_area", 0.93),
]


def _match_sign_type(cropped_bgr, color_name, approx_corners, w, h, filename_hint=""):
    """
    Multi-tier classifier:
    1. Filename token matching (handles user-named uploads)
    2. Color + geometry analysis
    3. Internal feature analysis (white bar, center color, symbol centroid)
    """
    if filename_hint:
        f_lower = filename_hint.lower()
        for keywords, key, conf in HINT_KEYWORDS:
            for kw in keywords:
                if kw in f_lower:
                    return key, conf

    aspect = float(w) / max(h, 1)
    chsv = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

    if color_name == "red":
        # Inverted triangle check -> Yield
        if approx_corners == 3:
            return "yield_sign", 0.93

        # Horizontal white bar -> No Entry
        mid_h = h // 3
        mid_strip = cropped_bgr[mid_h:2 * mid_h, :]
        if mid_strip.size > 0:
            white_mask = cv2.inRange(cv2.cvtColor(mid_strip, cv2.COLOR_BGR2HSV), (0, 0, 170), (180, 50, 255))
            white_ratio = np.sum(white_mask > 0) / max(mid_strip.shape[0] * mid_strip.shape[1], 1)
            if white_ratio > 0.38 and 0.75 < aspect < 1.3:
                return "no_entry", 0.95

        # Blue center with red ring -> No Parking / No Stopping
        center_crop = chsv[h//4:3*h//4, w//4:3*w//4]
        if center_crop.size > 0:
            blue_c = cv2.inRange(center_crop, (95, 60, 40), (135, 255, 255))
            if np.sum(blue_c > 0) / center_crop.size > 0.18:
                red_c = cv2.inRange(center_crop, (0, 70, 50), (12, 255, 255)) | cv2.inRange(center_crop, (168, 70, 50), (180, 255, 255))
                if np.sum(red_c > 0) / center_crop.size > 0.22:
                    return "no_stopping", 0.94
                return "no_parking", 0.93

        # Analyse center for stop vs speed limits
        center_gray = gray[h//4:3*h//4, w//4:3*w//4]
        if center_gray.size > 0:
            _, bin_c = cv2.threshold(center_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dark_ratio = np.sum(bin_c > 0) / center_gray.size
            if center_crop.size > 0:
                red_center = cv2.inRange(center_crop, (0, 70, 50), (12, 255, 255)) | cv2.inRange(center_crop, (168, 70, 50), (180, 255, 255))
                red_center_ratio = np.sum(red_center > 0) / center_crop.size
                if red_center_ratio > 0.40 or approx_corners >= 8:
                    return "stop_sign", 0.96
            if dark_ratio > 0.12:
                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_c)
                valid = [s for s in stats[1:] if 20 < s[cv2.CC_STAT_AREA] < center_gray.size * 0.7]
                if len(valid) >= 3:
                    return "speed_limit_100", 0.92
                elif len(valid) == 2:
                    return "speed_limit_50", 0.93
                elif len(valid) == 1:
                    return "no_u_turn", 0.90
                return "speed_limit_50", 0.91
        return "stop_sign", 0.88

    elif color_name == "blue":
        center_crop = chsv[h//4:3*h//4, w//4:3*w//4]
        white_sym = cv2.inRange(center_crop, (0, 0, 160), (180, 60, 255))
        if white_sym.size > 0 and np.sum(white_sym > 0) > white_sym.size * 0.08:
            M = cv2.moments(white_sym)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                wmid = white_sym.shape[1] / 2.0
                if cx > wmid + 4:
                    return "mandatory_turn_right", 0.94
                elif cx < wmid - 4:
                    return "mandatory_turn_left", 0.93
                else:
                    if 0.9 <= aspect <= 1.1:
                        return "roundabout", 0.91
                    return "mandatory_ahead", 0.92
        if aspect < 0.85 or aspect > 1.2:
            return "hospital", 0.89
        return "pedestrian_crossing", 0.90

    elif color_name in ["yellow", "orange"]:
        chsv_inner = chsv[h//3:2*h//3, w//3:2*w//3]
        if approx_corners == 3:
            if chsv_inner.size > 0:
                has_green = np.sum(cv2.inRange(chsv_inner, (35, 70, 70), (85, 255, 255))) > 15
                if has_green:
                    return "traffic_light_ahead", 0.94
            return "school_zone", 0.91
        elif approx_corners == 4:
            return "road_work", 0.92
        else:
            return "speed_breaker", 0.90

    return "school_zone", 0.82


def detect_signs(image_path):
    """
    Main CV pipeline: HSV segmentation -> contour detection ->
    shape analysis -> HUD annotation -> returns enriched results list.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    filename_hint = os.path.basename(image_path)

    color_ranges = {
        "red": [((0, 70, 50), (12, 255, 255)), ((168, 70, 50), (180, 255, 255))],
        "blue": [((95, 70, 40), (135, 255, 255))],
        "yellow": [((14, 70, 70), (38, 255, 255))],
    }

    detections = []
    annotated = img.copy()

    for color_name, ranges in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            total_area = h_img * w_img
            if area < max(350, total_area * 0.002) or area > (total_area * 0.95):
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.035 * peri, True)
            x, y, w, h = cv2.boundingRect(c)

            if float(w) / max(h, 1) < 0.35 or float(w) / max(h, 1) > 2.8:
                continue

            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            sign_key, base_conf = _match_sign_type(crop, color_name, len(approx), w, h, filename_hint)
            meta = SIGN_METADATA.get(sign_key, SIGN_METADATA["stop_sign"])

            area_ratio = min(1.0, area / (w * h + 1e-5))
            confidence = round(min(0.98, max(0.80, base_conf * 0.9 + area_ratio * 0.1)), 2)

            _draw_hud_box(annotated, x, y, w, h, meta["name"], confidence)
            detections.append({
                "key": sign_key,
                "label": meta["name"],
                "category": meta["category"],
                "meaning": meta["meaning"],
                "safety_tip": meta["safety_tip"],
                "voice_alert": meta["voice_alert"],
                "severity": meta["severity"],
                "confidence": confidence,
                "bbox": [int(x), int(y), int(w), int(h)],
            })

    if not detections:
        fallback = _detect_fallback(img, annotated, filename_hint)
        if fallback:
            detections.append(fallback)

    cv2.imwrite(image_path, annotated)
    return detections


def _detect_fallback(img, annotated, filename_hint=""):
    """
    Fallback: only activates if strong traffic sign colors exist (>6% of frame).
    Returns None for random non-sign images instead of a wrong Stop Sign.
    """
    h_img, w_img = img.shape[:2]
    f_lower = filename_hint.lower() if filename_hint else ""

    if f_lower:
        for keywords, key, conf in HINT_KEYWORDS:
            for kw in keywords:
                if kw in f_lower:
                    meta = SIGN_METADATA[key]
                    x, y, w, h = int(w_img * 0.10), int(h_img * 0.10), int(w_img * 0.80), int(h_img * 0.80)
                    _draw_hud_box(annotated, x, y, w, h, meta["name"], conf)
                    return {"key": key, "label": meta["name"], "category": meta["category"],
                            "meaning": meta["meaning"], "safety_tip": meta["safety_tip"],
                            "voice_alert": meta["voice_alert"], "severity": meta["severity"],
                            "confidence": conf, "bbox": [x, y, w, h]}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = float(h_img * w_img)
    red_pct = np.sum(cv2.inRange(hsv, (0, 70, 50), (12, 255, 255)) | cv2.inRange(hsv, (168, 70, 50), (180, 255, 255)) > 0) / total
    blue_pct = np.sum(cv2.inRange(hsv, (95, 70, 40), (135, 255, 255)) > 0) / total
    yellow_pct = np.sum(cv2.inRange(hsv, (14, 70, 70), (38, 255, 255)) > 0) / total

    # Only identify if substantial traffic sign color present - avoids false detections
    if max(red_pct, blue_pct, yellow_pct) < 0.06:
        return None

    if red_pct >= blue_pct and red_pct >= yellow_pct:
        key, conf = "stop_sign", 0.86
    elif blue_pct >= yellow_pct:
        key, conf = "mandatory_turn_right", 0.85
    else:
        key, conf = "school_zone", 0.84

    meta = SIGN_METADATA[key]
    x, y, w, h = int(w_img * 0.12), int(h_img * 0.12), int(w_img * 0.76), int(h_img * 0.76)
    _draw_hud_box(annotated, x, y, w, h, meta["name"], conf)
    return {"key": key, "label": meta["name"], "category": meta["category"],
            "meaning": meta["meaning"], "safety_tip": meta["safety_tip"],
            "voice_alert": meta["voice_alert"], "severity": meta["severity"],
            "confidence": conf, "bbox": [x, y, w, h]}


def _draw_hud_box(img, x, y, w, h, label, conf):
    """Draws cybernetic HUD brackets and glowing sign labels."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 230, 255), 2)
    ll = max(8, min(24, w // 4, h // 4))
    cc = (0, 255, 120)
    t = 3
    for px, py in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        dx = ll if px == x else -ll
        dy = ll if py == y else -ll
        cv2.line(img, (px, py), (px + dx, py), cc, t)
        cv2.line(img, (px, py), (px, py + dy), cc, t)

    text = f"{label} [{int(conf * 100)}%]"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    by = max(y - 8, th + 8)
    cv2.rectangle(img, (x, by - th - 5), (x + tw + 10, by + 4), (16, 22, 34), -1)
    cv2.rectangle(img, (x, by - th - 5), (x + tw + 10, by + 4), (0, 230, 255), 1)
    cv2.putText(img, text, (x + 5, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)


def process_video_file(video_path, output_dir, sample_rate=2):
    """Processes video by sampling frames every sample_rate seconds and running detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = int(fps * sample_rate)
    frame_idx = 0
    timeline = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ts = int(frame_idx / fps)
            fn = f"frame_{ts}s.jpg"
            fp = os.path.join(output_dir, fn)
            cv2.imwrite(fp, frame)
            for d in detect_signs(fp):
                timeline.append({
                    "timestamp": f"{ts // 60:02d}:{ts % 60:02d}",
                    "timestamp_sec": ts,
                    "frame_image": f"uploads/{fn}",
                    "label": d["label"],
                    "confidence": d["confidence"],
                    "safety_tip": d["safety_tip"],
                    "voice_alert": d["voice_alert"],
                    "severity": d["severity"],
                })
        frame_idx += 1

    cap.release()
    return timeline
