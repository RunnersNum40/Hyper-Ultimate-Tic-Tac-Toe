from util import MoveError
from itertools import product

import numpy as np

def direction_vectors(n: int):
    """Generate all posible direction vectors in n-dimensional space with unit length basis vectors.

    Args:
        n: Dimensionality of the space

    Returns:
        All direction vectors in n-dimensional space with unit components
    """
    # Yield all combinations of the unit vectors
    directions = np.array([direction for direction in product([1, 0, -1], repeat=n)], dtype=np.int32)
    for i in range(directions.shape[0], 0, -1):
        if (-directions[i-1]).tolist() in directions.tolist():
            directions = np.delete(directions, (i-1), axis=0)
    return directions

def check_move(board, position):
    """Check if a point on a board is in a winning line"""
    pass


class BoardMarker:
    """A marker placed on a board"""
    def __init__(self):
        raise NotImplementedError

    def value(self):
        """Value used to score games"""
        raise NotImplementedError

    def __str__(self):
        return f"Player: {self.value()}"


class Player(BoardMarker):
    """A marker that contains the name of the player that places it"""
    def __init__(self, name) -> None:
        self.name = name

    def value(self):
        return self.name


class HyperBoard(BoardMarker):
    """An n-dimensional hypercubic tic-tac-toe board

    Params:
        board: n-dimensional hypercubic np array of the board with integer player markers
        winner: integer marker of the winner if one exists
        shape: shape of the board (n length tuple of equal integer side lengths)
    """
    def __init__(self, side_length: int = 3, dimensions: int = 2) -> None:
        """A local board composed of a hypercube of positions.

        Args:
            side_length: side length of the board hypercube
            dimensions: number of dimensions the the board
            players: number of players
        """
        self.winner = None
        self.shape = tuple(side_length for _ in range(dimensions))
        self.board = np.empty(self.shape, dtype=object)

        self.ndims = dimensions
        self.side_length = side_length

        self._directions = direction_vectors(self.ndims)

    def place_marker(self, marker, position: tuple) -> bool:
        """Place a marker on the board and return if it wins

        Args:
            player: Board Marker object of the player
            position: n-dimensional position in the board
        Returns:
            True if the move wins the board, False if the move does not win the board.
        """
        # Allow integer players
        if type(marker) is int:
            marker = Player(marker)
        # Check that the move is legal
        if self.board[position] is not None:
            raise MoveError(f"{self.board[position]} already in position")
        # Place the marker
        self.board[position] = marker
        # Check if the move wins
        position = np.array(position, dtype=np.int32)
        # Vary the point placed in each possible direction
        for direction in self._directions:
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
                line = [marker.value() if marker is not None else None for marker in line]
                # Check if all markers are the same
                if None not in line and len(set(line)) == 1:
                    # The placement wins the board
                    self.winner = marker
                    return True
        # Not a winning placement
        return False

    def value(self):
        return self.winner

    def __str__(self):
        if self.board.ndim <= 3:
            return str(self.board)
        else:
            return f"{self.board.ndim}-dimensional board with side length of {self.shape[0]}"


class StandardBoard(HyperBoard):
    def __init__(self):
        super().__init__(side_length = 3, dimensions = 2)


if __name__ == '__main__':
    marker = Player(1)
    player = StandardBoard()
    player.place_marker(marker, (0, 0))
    player.place_marker(marker, (1, 0))
    player.place_marker(marker, (2, 0))

    # Test if a board allows placing on a spot twice
    board = HyperBoard(1, 1)
    assert board.place_marker(marker, (0, ))
    try:
        board.place_marker(marker, (0, ))
    except MoveError:
        pass

    # Test a normal board for row, column, and diagonal wins
    # Row
    board = HyperBoard(3, 2)
    assert not board.place_marker(player, (0, 1))
    assert not board.place_marker(player, (1, 1))
    assert     board.place_marker(player, (2, 1))
    # Column
    board = HyperBoard(3, 2)
    assert not board.place_marker(player, (1, 0))
    assert not board.place_marker(player, (1, 1))
    assert     board.place_marker(player, (1, 2))
    # Diagonal
    board = HyperBoard(3, 2)
    assert not board.place_marker(player, (0, 0))
    assert not board.place_marker(player, (1, 1))
    assert     board.place_marker(player, (2, 2))
    # Anti-diagonal
    board = HyperBoard(3, 2)
    assert not board.place_marker(player, (2, 0))
    assert not board.place_marker(player, (1, 1))
    assert     board.place_marker(player, (0, 2))
    # Test a 3D board
    board = HyperBoard(3, 3)
    assert not board.place_marker(player, (0, 1, 1))
    assert not board.place_marker(player, (1, 1, 1))
    assert     board.place_marker(player, (2, 1, 1))
    board = HyperBoard(3, 3)
    assert not board.place_marker(player, (1, 0, 1))
    assert not board.place_marker(player, (1, 1, 1))
    assert     board.place_marker(player, (1, 2, 1))
    board = HyperBoard(3, 3)
    assert not board.place_marker(player, (1, 1, 0))
    assert not board.place_marker(player, (1, 1, 1))
    assert     board.place_marker(player, (1, 1, 2))
    board = HyperBoard(3, 3)
    assert not board.place_marker(player, (2, 0, 0))
    assert not board.place_marker(player, (1, 1, 0))
    assert     board.place_marker(player, (0, 2, 0))
    board = HyperBoard(3, 3)
    assert not board.place_marker(player, (0, 2, 0))
    assert not board.place_marker(player, (0, 1, 1))
    assert     board.place_marker(player, (0, 0, 2))
    board = HyperBoard(3, 3)
    assert not board.place_marker(player, (0, 0, 2))
    assert not board.place_marker(player, (1, 0, 1))
    assert     board.place_marker(player, (2, 0, 0))
    board = HyperBoard(3, 3)
    assert not board.place_marker(player, (0, 0, 0))
    assert not board.place_marker(player, (1, 1, 1))
    assert     board.place_marker(player, (2, 2, 2))
    board = HyperBoard(3, 3)
    assert not board.place_marker(player, (2, 2, 2))
    assert not board.place_marker(player, (1, 1, 1))
    assert     board.place_marker(player, (0, 0, 0))