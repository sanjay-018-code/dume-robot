
import cv2


class CameraSource:
    """
    Camera-independent frame source.

    The vision system only asks for frames; it does not
    need to know which camera hardware provides them.
    """

    def __init__(self, source=0, backend=None):
        self.source = source

        if backend is None and isinstance(source, int):
            backend = cv2.CAP_DSHOW

        if backend is None:
            self.capture = cv2.VideoCapture(source)
        else:
            self.capture = cv2.VideoCapture(source, backend)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {source}"
            )

    def read(self):
        """Return (success, frame), just like OpenCV."""
        return self.capture.read()

    def is_opened(self):
        return self.capture.isOpened()

    def release(self):
        self.capture.release()