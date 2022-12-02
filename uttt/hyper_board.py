from util import MoveError
from itertools import product

import numpy as np


def ternary_list(n):
    if n == 0:
        return [0]
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(r)
    return nums

def direction_vectors(n: int):
    """Generate all posible direction vectors in n-dimensional space with unit length basis vectors.

    Args:
        n: Dimensionality of the space

    Returns:
        All direction vectors in n-dimensional space with unit components
    """
    # Yield all combinations of the unit vectors
    zero = tuple(0 for _ in range(n))
    return np.array([direction for direction in product([-1, 0, 1], repeat=n) if direction != zero], dtype=np.int32)


class HyperBoard:
    """An n-dimensional hypercubic tic-tac-toe board

    Params:
        board: n-dimensional hypercubic np array of the board with integer player markers
        winner: integer marker of the winner if one exists
        shape: shape of the board (n length tuple of equal integer side lengths)
    """
    def __init__(self, side_length: int = 3, dimensions: int = 2, players: int = 2) -> None:
        """A local board composed of a hypercube of positions.

        Args:
            side_length: side length of the board hypercube
            dimensions: number of dimensions the the board
            players: number of players
        """
        self.winner = None
        self.shape = tuple(side_length for _ in range(dimensions))
        self.board = np.full(self.shape, np.nan)

        self.ndims = dimensions
        self.side_length = side_length

        self.directions = direction_vectors(self.ndims)

    def place_marker(self, player: int, position: tuple) -> bool:
        """Place a marker on the board and return if it wins

        Args:
            player: Integer marker of the player
            position: n-dimensional position in the board
        Returns:
            True if the move wins the board, False if the move does not win the board.
        """
        # Check that the move is legal
        if not np.isnan(self.board[position]):
            raise MoveError(f"{self.board[position]} already in position")
        # Place the marker
        self.board[position] = player
        # Check if the move wins
        position = np.array(position, dtype=np.int32)
        # Vary the point placed in each possible direction
        for direction in self.directions:
            # Find the range possible in the given direction
            test_min = -self.ndims
            test_max = self.ndims+1
            for dx, x in zip(direction, position):
                if dx == 1:
                    test_min = max(test_min, -x)
                    test_max = min(test_max, self.side_length-x)
                if dx == -1:
                    test_min = max(test_min, 1-(self.side_length-x))
                    test_max = min(test_max, x+1)
            # If the line is long enough to win
            if test_max-test_min == self.side_length:
                # Find all points along the line
                line = [self.board[tuple(position+i*direction)] for i in range(test_min, test_max)]
                # Check if all markers are the same
                if len(set(line)) == 1:
                    # The placement wins the board
                    self.winner = player
                    return True
        # Not a winning placement
        return False

    def __str__(self):
        if self.board.ndim <= 3:
            return str(self.board)
        else:
            return f"{self.board.ndim}-dimensional board with side length of {self.shape[0]}"