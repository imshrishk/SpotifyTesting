import cv2
import numpy as np
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

class SpotifyGestureController:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.wCam, self.hCam = 640, 480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volRange = self.volume.GetVolumeRange()
        self.minVol, self.maxVol = self.volRange[0], self.volRange[1]
        
        scope = "user-read-playback-state,user-modify-playback-state"
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                            client_secret=client_secret,
                                                            redirect_uri=redirect_uri,
                                                            scope=scope))
        
        self.volume_control_active = False
        self.prev_vol = self.volume.GetMasterVolumeLevel()
        self.calibrated = False
        self.min_hand_size = float('inf')
        self.max_hand_size = 0
        self.prev_hand_center = None
        self.palm_state = None
        self.prev_palm_state = None
        self.action = None
        self.last_action_time = 0
        self.action_cooldown = 0  # cooldown between actions
        
        self.swipe_threshold = 100  # Threshold for deliberate swipes
        self.swipe_start = None
        self.swipe_direction = None

    def find_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list

    def calculate_hand_size(self, lm_list):
        if len(lm_list) == 0:
            return None
        wrist = lm_list[0]
        middle_finger_tip = lm_list[12]
        hand_size = math.hypot(middle_finger_tip[1] - wrist[1], middle_finger_tip[2] - wrist[2])
        
        if not self.calibrated:
            self.min_hand_size = min(self.min_hand_size, hand_size)
            self.max_hand_size = max(self.max_hand_size, hand_size)
            if self.max_hand_size - self.min_hand_size > 100:
                self.calibrated = True
        
        return hand_size

    def calculate_palm_state(self, lm_list):
        if len(lm_list) < 21:
            return None
        
        thumb_tip = lm_list[4]
        index_tip = lm_list[8]
        middle_tip = lm_list[12]
        ring_tip = lm_list[16]
        pinky_tip = lm_list[20]
        wrist = lm_list[0]
        
        fingers_extended = sum([
            thumb_tip[1] < wrist[1],
            index_tip[2] < lm_list[6][2],
            middle_tip[2] < lm_list[10][2],
            ring_tip[2] < lm_list[14][2],
            pinky_tip[2] < lm_list[18][2]
        ])
        
        return "open" if fingers_extended >= 4 else "closed"

    def calculate_hand_center(self, lm_list):
        if len(lm_list) < 21:
            return None
        
        x_coords = [lm[1] for lm in lm_list]
        y_coords = [lm[2] for lm in lm_list]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return (int(center_x), int(center_y))

    def get_active_device(self):
        devices = self.sp.devices()
        if devices['devices']:
            for device in devices['devices']:
                if device['is_active']:
                    return device['id']
            return devices['devices'][0]['id']
        return None

    def detect_gestures(self, lm_list):
        if len(lm_list) == 0:
            return None

        hand_size = self.calculate_hand_size(lm_list)
        if not hand_size:
            return None

        self.palm_state = self.calculate_palm_state(lm_list)
        hand_center = self.calculate_hand_center(lm_list)

        # Check if hand is within frame
        h, w, _ = self.img.shape
        if not (0 <= hand_center[0] <= w and 0 <= hand_center[1] <= h):
            return None

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - self.last_action_time < self.action_cooldown:
            return None

        if self.palm_state != self.prev_palm_state:
            if self.palm_state == "open":
                self.action = "Play"
                self.handle_play()
            elif self.palm_state == "closed":
                self.action = "Pause"
                self.handle_pause()
            self.prev_palm_state = self.palm_state
            self.last_action_time = current_time
            return self.action

        # New swipe gesture detection
        if self.swipe_start is None:
            self.swipe_start = hand_center
        else:
            dx = hand_center[0] - self.swipe_start[0]
            if abs(dx) > self.swipe_threshold:
                if dx > 0:
                    self.action = "Next Track"
                    self.handle_next_track()
                else:
                    self.action = "Previous Track"
                    self.handle_previous_track()
                self.swipe_start = None
                self.last_action_time = current_time
                return self.action

        thumb_tip = lm_list[4]
        index_tip = lm_list[8]
        volume_length = math.hypot(index_tip[1] - thumb_tip[1], index_tip[2] - thumb_tip[2]) / hand_size
        if 0.1 < volume_length < 0.5:
            self.volume_control_active = True
            self.action = "Volume Control"
            self.handle_volume_control(volume_length)
            self.last_action_time = current_time
            return self.action

        self.volume_control_active = False
        return None

    def handle_volume_control(self, length):
        vol = np.interp(length, [0.1, 0.5], [self.minVol, self.maxVol])
        self.volume.SetMasterVolumeLevel(vol, None)
        self.prev_vol = vol
        
        volPer = np.interp(vol, [self.minVol, self.maxVol], [0, 100])
        cv2.putText(self.img, f'Vol: {int(volPer)}%', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def handle_play(self):
        device_id = self.get_active_device()
        if device_id:
            try:
                self.sp.start_playback(device_id=device_id)
            except spotipy.exceptions.SpotifyException as e:
                print(f"Error starting playback: {e}")

    def handle_pause(self):
        device_id = self.get_active_device()
        if device_id:
            try:
                self.sp.pause_playback(device_id=device_id)
            except spotipy.exceptions.SpotifyException as e:
                print(f"Error pausing playback: {e}")

    def handle_next_track(self):
        device_id = self.get_active_device()
        if device_id:
            try:
                self.sp.next_track(device_id=device_id)
            except spotipy.exceptions.SpotifyException as e:
                print(f"Error skipping to next track: {e}")

    def handle_previous_track(self):
        device_id = self.get_active_device()
        if device_id:
            try:
                self.sp.previous_track(device_id=device_id)
            except spotipy.exceptions.SpotifyException as e:
                print(f"Error skipping to previous track: {e}")

    def run(self):
        while True:
            success, self.img = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            self.img = self.find_hands(self.img)
            lm_list = self.find_position(self.img)
            action = self.detect_gestures(lm_list)
            
            # Always display the last action
            cv2.putText(self.img, f'Last Action: {self.action if self.action else "None"}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display system volume
            current_vol = self.volume.GetMasterVolumeLevelScalar()
            sys_vol_percent = int(current_vol * 100)
            cv2.putText(self.img, f'Sys Vol: {sys_vol_percent}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(self.img, "Open palm: Play | Closed palm: Pause", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(self.img, "Swipe left/right: Previous/Next track", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(self.img, "Pinch: Volume control", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Spotify Gesture Control", self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    client_id = ''
    client_secret = ''
    redirect_uri = 'http://localhost:8888/callback'

    controller = SpotifyGestureController(client_id, client_secret, redirect_uri)
    controller.run()
