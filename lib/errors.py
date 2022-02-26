class InvalidMoveError(Exception):
    """
    Error to raise when a move is invalid.
    """
    def __init__(self):
        self.message = "Your move is a invalid move!"
