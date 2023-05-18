class IncompatibleDimensionsException(Exception):
    def __init__(self):
        self.message = "Previous layer shape is incompatible with this layer's shape!"
        super().__init__(self.message)

