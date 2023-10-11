from configs.simple_classifier_cfg import cfg


class Classifier:
    def __call__(self, height):
        """returns confidence of belonging to the class of basketball players"""
        return height / cfg.max_height
