#! python3
import pyautogui, sys
def run_mouse_tracking():
    print('Press Ctrl-C to quit.')
    try:
        while True:
            x, y = pyautogui.position()
            positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            print(positionStr, end='')
            print('\b' * len(positionStr), end='', flush=True)
    except KeyboardInterrupt:
        print('\n')


def run_mouse_distance():
    print('Press Ctrl-C to quit.')
    try:
        y_sum = 0
        x_sum = 0
        while True:
            x, y = pyautogui.position()
            pyautogui.moveTo(800,450)
            y_sum += y - 450
            x_sum += x - 800
            positionStr = 'distance y: ' + str(y_sum).rjust(4) + 'distance x: ' + str(x_sum).rjust(4)
            print(positionStr, end='')
            x, y = pyautogui.position()
            pyautogui.moveTo(800,450)
            y_sum += y - 450
            x_sum += x - 800
            print('\b' * len(positionStr), end='', flush=True)
            x, y = pyautogui.position()
            pyautogui.moveTo(800,450)
            y_sum += y - 450
            x_sum += x - 800
    except KeyboardInterrupt:
        print('\n')

run_mouse_tracking()
