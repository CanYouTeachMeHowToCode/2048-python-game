## pytest tests for board.py

import copy
import pytest

from game2048.board import Board


def make_board(rows, score=0):
    """Build a Board with a fixed size and layout, bypassing the random init tile."""
    size = len(rows)
    board = Board(size)
    board.board = [row[:] for row in rows]
    board.score = score
    return board


def test_init_board_has_exactly_one_starting_tile():
    board = Board(4)
    assert board.size == 4
    assert len(board.board) == 4
    assert all(len(row) == 4 for row in board.board)

    nonEmpty = [(i, j) for i in range(4) for j in range(4) if board.board[i][j] != 0]
    assert len(nonEmpty) == 1
    i, j = nonEmpty[0]
    assert board.board[i][j] in (2, 4)
    assert board.score == 0


def test_add_new_tile_fills_an_empty_cell():
    board = make_board([
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    addIndex, addNum = board.addNewTile()

    assert addNum in (2, 4)
    assert board.board[addIndex[0]][addIndex[1]] == addNum
    nonEmpty = sum(1 for row in board.board for x in row if x != 0)
    assert nonEmpty == 2


def test_is_same_board():
    board = Board(4)
    b1 = [[1, 2], [3, 4]]
    b2 = [[1, 2], [3, 4]]
    b3 = [[1, 2], [3, 5]]
    assert board.isSameBoard(b1, b2) is True
    assert board.isSameBoard(b1, b3) is False


def test_move_left_merges_and_shifts():
    board = make_board([
        [2, 2, 0, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    canMove = board.moveLeft()
    assert canMove is True
    assert board.board[0] == [4, 4, 0, 0]
    assert board.score == 4


def test_move_left_merges_only_once_per_pair():
    board = make_board([
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    board.moveLeft()
    assert board.board[0] == [4, 4, 0, 0]
    assert board.score == 8


def test_move_left_returns_false_when_no_move_possible():
    board = make_board([
        [2, 4, 8, 16],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    canMove = board.moveLeft()
    assert canMove is False
    assert board.board[0] == [2, 4, 8, 16]


def test_transpose():
    board = make_board([
        [1, 2],
        [3, 4],
    ])
    board.transpose()
    assert board.board == [[1, 3], [2, 4]]


def test_mirror():
    board = make_board([
        [1, 2],
        [3, 4],
    ])
    board.mirror()
    assert board.board == [[2, 1], [4, 3]]


def test_rotate90_clockwise():
    board = make_board([
        [1, 2],
        [3, 4],
    ])
    board.rotate90Clockwise()
    assert board.board == [[3, 1], [4, 2]]


def test_rotate180():
    board = make_board([
        [1, 2],
        [3, 4],
    ])
    board.rotate180()
    assert board.board == [[4, 3], [2, 1]]


def test_rotate90_counterclockwise():
    board = make_board([
        [1, 2],
        [3, 4],
    ])
    board.rotate90CounterClockwise()
    assert board.board == [[2, 4], [1, 3]]


def test_rotate90_clockwise_then_counterclockwise_is_identity():
    original = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    board = make_board(original)
    board.rotate90Clockwise()
    board.rotate90CounterClockwise()
    assert board.board == original


def test_move_down_merges_column():
    board = make_board([
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [4, 0, 0, 0],
    ])
    canMove = board.moveDown()
    assert canMove is True
    expectedCol0 = [0, 0, 4, 4]
    assert [board.board[i][0] for i in range(4)] == expectedCol0
    assert board.score == 4


def test_move_up_merges_column():
    board = make_board([
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [4, 0, 0, 0],
    ])
    canMove = board.moveUp()
    assert canMove is True
    expectedCol0 = [4, 4, 0, 0]
    assert [board.board[i][0] for i in range(4)] == expectedCol0
    assert board.score == 4


def test_move_right_merges_and_shifts():
    board = make_board([
        [4, 0, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    canMove = board.moveRight()
    assert canMove is True
    assert board.board[0] == [0, 0, 4, 4]
    assert board.score == 4


def test_reaches2048_false_when_below_threshold():
    board = make_board([
        [1024, 0],
        [0, 0],
    ])
    assert board.reaches2048() is False


def test_reaches2048_true_when_tile_at_or_above_threshold():
    board = make_board([
        [2048, 0],
        [0, 0],
    ])
    assert board.reaches2048() is True


def test_game_over_false_when_a_move_is_available():
    board = make_board([
        [2, 4, 8, 16],
        [4, 8, 16, 32],
        [8, 16, 32, 64],
        [16, 32, 2, 2],
    ])
    assert board.GameOver() is False
    # GameOver should restore the board and score after probing moves
    assert board.board[3][2:] == [2, 2]
    assert board.score == 0


def test_game_over_true_when_no_moves_and_no_merges_possible():
    board = make_board([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ])
    assert board.GameOver() is True
