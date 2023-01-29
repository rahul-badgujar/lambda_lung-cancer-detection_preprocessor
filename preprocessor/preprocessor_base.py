import cv2

class Preprocessor:

    def process(self,ip_img_from,op_img_at)->None:
        raise NotImplemented("Preprocessor must implement process")

