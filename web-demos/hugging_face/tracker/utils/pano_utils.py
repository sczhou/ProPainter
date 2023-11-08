import numpy as np
from threading import Lock


class ID2RGBConverter:
    def __init__(self):
        self.all_id = []
        self.obj_to_id = {}
        self.lock = Lock()

    def _id_to_rgb(self, id: int):
        rgb = np.zeros((3, ), dtype=np.uint8)
        for i in range(3):
            rgb[i] = id % 256
            id = id // 256
        return rgb

    def convert(self, obj: int):
        with self.lock:
            if obj in self.obj_to_id:
                id = self.obj_to_id[obj]
            else:
                while True:
                    id = np.random.randint(255, 256**3)
                    if id not in self.all_id:
                        break
                self.obj_to_id[obj] = id
                self.all_id.append(id)

        return id, self._id_to_rgb(id)
