from enum import IntFlag


class DrawTaps(IntFlag):
    LEFT = 1
    RIGHT = 2
    NONE = ~(LEFT | RIGHT)
    BOTH = LEFT & RIGHT