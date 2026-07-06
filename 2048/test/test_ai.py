## pytest tests for AI.py

import pytest

from game2048.board import Board
from game2048.AI import AI


def make_board(rows, score=0):
    """Build a Board with a fixed size and layout, bypassing the random init tile."""
    size = len(rows)
    board = Board(size)
    board.board = [row[:] for row in rows]
    board.score = score
    return board


def make_ai(rows, level=2, score=0):
    board = make_board(rows, score=score)
    return AI(board, level)


def test_evaluate_uses_weight_board():
    ai = make_ai([
        [2, 0],
        [0, 0],
    ])
    assert ai.evaluate() == 2 * ai.weightBoard[0][0]


def test_evaluate_sums_weighted_tiles():
    ai = make_ai([
        [2, 4],
        [0, 8],
    ])
    expected = (
        2 * ai.weightBoard[0][0]
        + 4 * ai.weightBoard[0][1]
        + 8 * ai.weightBoard[1][1]
    )
    assert ai.evaluate() == expected


def test_get_legal_moves_excludes_blocked_directions():
    # only row 0 is filled with a mergeable pair: Up is blocked (already at
    # the top), Down/Left/Right all change the board.
    ai = make_ai([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    legalMoves = ai.getLegalMoves()
    assert sorted(legalMoves) == [1, 2, 3]  # Down, Left, Right


def test_get_legal_moves_empty_when_board_is_stuck():
    ai = make_ai([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ])
    assert ai.getLegalMoves() == []


def test_perform_action_applies_move_and_adds_new_tile():
    ai = make_ai([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    ai.performAction(2)  # Left

    # the merge always lands in the leftmost column; the newly added tile
    # can land on any empty cell, so only check what performAction guarantees.
    assert ai.GameBoard.board[0][0] == 4
    assert ai.GameBoard.score == 4
    nonEmpty = sum(1 for row in ai.GameBoard.board for x in row if x != 0)
    assert nonEmpty == 2  # merged tile + the newly added tile


def test_add_new_num_places_requested_value():
    ai = make_ai([
        [0, 0],
        [0, 0],
    ])
    ai.addNewNum(((1, 1), 4))
    assert ai.GameBoard.board[1][1] == 4


def test_add_new_num_rejects_non_empty_tile():
    ai = make_ai([
        [2, 0],
        [0, 0],
    ])
    with pytest.raises(AssertionError):
        ai.addNewNum(((0, 0), 4))


def test_get_important_tiles_returns_highest_weight_empty_tiles():
    ai = make_ai([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    importantTiles = ai.getImporantTiles(3)
    allTiles = [(i, j) for i in range(4) for j in range(4)]
    expected = sorted(
        allTiles, key=lambda coord: ai.weightBoard[coord[0]][coord[1]], reverse=True
    )[:3]
    assert importantTiles == expected
