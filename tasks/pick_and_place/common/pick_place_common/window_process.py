"""The plant's windows, in their own process: the decision window and the
live obstacle window. Showing an image with OpenCV costs ~10-25 ms under
WSLg, and doing it in the plant's process, even on another thread, stalled
physics steps up to 200 ms. The plant composes the images and sends them
over a pipe. Imports only OpenCV.
"""
import cv2


def run(conn, titles):
    """Child process entry: create the windows, reply True/False, then show
    each (title, image) received, newest per window, until None or EOF."""
    # OpenCV's worker threads spin between calls: with frames coming in, ~14 cores busy.
    cv2.setNumThreads(1)
    try:
        for title in titles:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        conn.send(True)
    except Exception:
        conn.send(False)
        return
    try:
        while True:
            if conn.poll(0.05):
                latest = {}
                msg = conn.recv()
                while msg is not None:
                    latest[msg[0]] = msg[1]
                    if not conn.poll():
                        break
                    msg = conn.recv()
                for title, img in latest.items():
                    cv2.imshow(title, img)
                if msg is None:
                    break
            cv2.waitKey(1)
    except (EOFError, OSError, KeyboardInterrupt):
        pass
    cv2.destroyAllWindows()
