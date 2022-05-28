import numpy as np

def walls_to_text(walls, nrows=3, ncols=3, row=0, col=0):
    txt = ''
    for row in walls[row:row+nrows,col:col+ncols]:
        for cell in row:
            mark = '-' if cell == 1 else 'x'
            txt += mark + ' '
        txt += '\n'

    return txt

def fires_to_text(fires, nrows=3, ncols=3, row=0, col=0):
    txt = ''
    for row in fires[:nrows,:ncols]:
        for cell in row:
            txt += str(cell) + ' '
        txt += '\n'

    return txt

def euclidian_cost(position, goal) -> np.float:
    return np.linalg.norm(goal - position)

def is_wall(direction, walls) -> bool:
    return False

def is_fire(direction, fires) -> bool:
    return False
