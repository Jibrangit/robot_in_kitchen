from typing import List, Tuple

def plot_line_low(x0, y0, x1, y1) -> List[Tuple[int]]:
    '''
        Lines with low slopes, where X changes more frequently than Y
    '''
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0
    coordinates = []

    for x in range(x0, x1 + 1):
        coordinates.append((x, y))
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy


    return coordinates

def plot_line_high(x0, y0, x1, y1) -> List[Tuple[int]]:
    '''
        Lines with steep slopes, where Y changes more frequently than X
    '''
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    coordinates = []

    for y in range(y0, y1 + 1):
        coordinates.append((x, y))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx

    return coordinates

def plot_line(x0, y0, x1, y1) -> List[Tuple[int]]:
    '''
        Given 2 integer coordinates, retrieve all integer coordinates lying on the line connecting the coordinates.
    '''
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            plot = plot_line_low(x1, y1, x0, y0)
            plot.reverse()
            return plot
        else:
            return plot_line_low(x0, y0, x1, y1)
    else:
        if y0 > y1:
            plot = plot_line_high(x1, y1, x0, y0)
            plot.reverse()
            return plot
        else:
            return plot_line_high(x0, y0, x1, y1)
