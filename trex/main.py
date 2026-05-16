from mss import MSS
import numpy as np
import cv2
import pyautogui
import time

monitor = {"top":250, "left": 300, "width": 1000, "height": 200}

DINO_X = 200
GROUND_Y = 160
DETECTOR_H = 50

SECONDS_TO_MAX = 116
INITIAL_SPEED_PX = 390
MAX_SPEED_PX = 600
JUMP_DURATION_MIN = 0.15
JUMP_DURATION_MAX = 0.5
JUMP_COOLDOWN = 0.01

GAMEOVER_X = 350
GAMEOVER_Y = 50
GAMEOVER_W = 200
GAMEOVER_H = 40
GAMEOVER_THRESHOLD = 400

game_start = None
first_obstacle_seen = False
last_jump = 0

def find_all_obstacles(binary, search_x_start, search_x_end, dy, dh):
    roi = binary[dy:dy+dh, search_x_start:search_x_end]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 15:
            continue
        abs_x = search_x_start + x
        abs_x2 = abs_x + w
        obstacles.append((abs_x, abs_x2, w, h))
    obstacles.sort(key=lambda o: o[0])
    return obstacles


def should_jump(obstacles, dino_x, current_speed_px, jump_duration):
    if not obstacles:
        return False

    distance_traveled = current_speed_px * jump_duration

    for x1, x2, w, h in obstacles:
        x1_after = x1 - distance_traveled
        x2_after = x2 - distance_traveled

        # этот кактус будет на месте динозавра после приземления
        if x1_after < dino_x + 50 and x2_after > dino_x - 10:
            # прыгаем когда первый кактус входит в зону прыжка
            first_distance = obstacles[0][0] - dino_x
            return first_distance <= current_speed_px * jump_duration

    first_distance = obstacles[0][0] - dino_x
    return first_distance <= current_speed_px * jump_duration


time.sleep(3)
pyautogui.press("space")

with MSS() as sct:
    while True:
        img = np.array(sct.grab(monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        now = time.time()
        dy = GROUND_Y - DETECTOR_H

        elapsed = time.time() - game_start if game_start else 0
        progress = min(elapsed / SECONDS_TO_MAX, 1.0)
        current_speed_px = INITIAL_SPEED_PX + progress * (MAX_SPEED_PX - INITIAL_SPEED_PX)
        jump_duration = JUMP_DURATION_MIN + progress * (JUMP_DURATION_MAX - JUMP_DURATION_MIN)
        jump_distance = current_speed_px * jump_duration

        # Детект Game Over
        gameover_zone = binary[GAMEOVER_Y:GAMEOVER_Y+GAMEOVER_H,
                               GAMEOVER_X:GAMEOVER_X+GAMEOVER_W]
        gameover_pixels = cv2.countNonZero(gameover_zone)
        is_gameover = gameover_pixels > GAMEOVER_THRESHOLD

        if is_gameover:
            time.sleep(1)
            pyautogui.press("space")
            game_start = None
            first_obstacle_seen = False
            last_jump = 0

        # Находим все препятствия
        obstacles = find_all_obstacles(binary, DINO_X + 20, DINO_X + 500, dy, DETECTOR_H)

        if obstacles and not first_obstacle_seen:
            first_obstacle_seen = True
            game_start = time.time()

        if not is_gameover and should_jump(obstacles, DINO_X, current_speed_px, jump_duration):
            if now - last_jump > JUMP_COOLDOWN:
                pyautogui.press("space")
                last_jump = now

        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for i, (x1, x2, w, h) in enumerate(obstacles):
            # показываем где кактус будет после приземления
            x1_after = int(x1 - jump_distance)
            x2_after = int(x2 - jump_distance)
            color = (0, 0, 255) if i == 0 else (0, 165, 255)
            cv2.rectangle(debug, (x1, dy), (x2, dy+DETECTOR_H), color, 2)

        if obstacles:
            first_distance = obstacles[0][0] - DINO_X
            color = (0, 0, 255) if first_distance <= jump_distance else (0, 255, 0)
            cv2.line(debug, (DINO_X, GROUND_Y - 25), (obstacles[0][0], GROUND_Y - 25), color, 2)
            cv2.putText(debug,
                        f"dist: {first_distance:.0f}  jump_dist: {jump_distance:.0f}  speed: {current_speed_px:.0f}  n: {len(obstacles)}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.putText(debug, f"no obstacle  speed: {current_speed_px:.0f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.rectangle(debug, (DINO_X + 20, dy), (DINO_X + 500, dy+DETECTOR_H), (255, 165, 0), 1)
        cv2.rectangle(debug,
                      (GAMEOVER_X, GAMEOVER_Y),
                      (GAMEOVER_X+GAMEOVER_W, GAMEOVER_Y+GAMEOVER_H),
                      (255, 0, 255), 1)

        cv2.imshow("debug", debug)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break