# test board.py
from board import Board
import numpy as np

def test_initial_board():
    testBoard = Board()
    assert testBoard.size == 4
    assert testBoard.empty == 0
    assert testBoard.score == 0
    assert testBoard.directionList == ["Up", "Down", "Left", "Right"]
    assert testBoard.board.shape == (4, 4)
    assert testBoard.board.dtype == int

    # test board with another size
    testBoard2 = Board(size=6)
    assert testBoard2.size == 6
    assert testBoard2.empty == 0
    assert testBoard2.score == 0
    assert testBoard2.directionList == ["Up", "Down", "Left", "Right"]
    assert testBoard2.board.shape == (6, 6)
    assert testBoard2.board.dtype == int

def createTestBoard():
    testBoard = Board()
    testBoard.board = np.array([[4, 2, 4, 2],
                                [4, 2, 4, 8],
                                [64, 128, 8, 16],
                                [8, 256, 128, 32]])
    return testBoard

def test_transpose():
    testBoard = createTestBoard()
    testBoard.transpose()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board, 
                          np.array([[4, 4, 64, 8], 
                                    [2, 2, 128, 256], 
                                    [4, 4, 8, 128], 
                                    [2, 8, 16, 32]]))

def test_mirror():
    testBoard = createTestBoard()
    testBoard.mirror()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board, 
                          np.array([[2, 4, 2, 4], 
                                    [8, 4, 2, 4], 
                                    [16, 8, 128, 64], 
                                    [32, 128, 256, 8]]))
    
def test_rotate90Clockwise():
    testBoard = createTestBoard()
    testBoard.rotate90Clockwise()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board,
                            np.array([[8, 64, 4, 4],
                                      [256, 128, 2, 2],
                                      [128, 8, 4, 4],
                                      [32, 16, 8, 2]]))

def test_rotate180():
    testBoard = createTestBoard()
    testBoard.rotate180()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board,
                          np.array([[32, 128, 256, 8],
                                    [16, 8, 128, 64],
                                    [8, 4, 2, 4],
                                    [2, 4, 2, 4]]))

def test_rotate90CounterClockwise():
    testBoard = createTestBoard()
    testBoard.rotate90CounterClockwise()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board,
                          np.array([[2, 8, 16, 32],
                                    [4, 4, 8, 128],
                                    [2, 2, 128, 256],
                                    [4, 4, 64, 8]]))

def test_moveDown():
    testBoard = createTestBoard()
    testBoard.moveDown()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board,
                          np.array([[0, 0, 0, 2],
                                    [8, 4, 8, 8],
                                    [64, 128, 8, 16],
                                    [8, 256, 128, 32]]))

def test_moveUp():
    testBoard = createTestBoard()
    testBoard.moveUp()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board,
                          np.array([[8, 4, 8, 2],
                                    [64, 128, 8, 8],
                                    [8, 256, 128, 16],
                                    [0, 0, 0, 32]]))

def test_moveLeft():
    testBoard = createTestBoard()
    testBoard.moveLeft()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board,
                          np.array([[4, 2, 4, 2],
                                    [4, 2, 4, 8],
                                    [64, 128, 8, 16],
                                    [8, 256, 128, 32]]))

def test_moveRight():
    testBoard = createTestBoard()
    testBoard.moveRight()
    assert testBoard.board.shape == (4, 4)
    assert np.array_equal(testBoard.board,
                          np.array([[4, 2, 4, 2],
                                    [4, 2, 4, 8],
                                    [64, 128, 8, 16],
                                    [8, 256, 128, 32]]))

def test_reaches2048():
    testBoard = createTestBoard()
    assert not testBoard.reaches2048() 
    # a board example that reaches 2048
    testBoard2 = Board()
    testBoard2.board = np.array([[4, 2, 4, 2],
                                 [4, 2, 4, 8],
                                 [64, 128, 8, 16],
                                 [8, 256, 128, 2048]])
    assert testBoard2.reaches2048() 
                                 

def test_getLargestTileNumber():
    testBoard = createTestBoard()
    assert testBoard.getLargestTileNumber() == 256

# a 2048 game board that game over (cannot move in any direction)
def createTestBoard2():
    testBoard = Board()
    testBoard.board = np.array([[2, 4, 2, 4],
                                [4, 16, 4, 2],
                                [2, 4, 8, 4],
                                [4, 2, 4, 2]])
    return testBoard

def test_gameOver():
    testBoard = createTestBoard2()
    assert testBoard.gameOver()
    testBoard2 = createTestBoard()
    assert not testBoard2.gameOver()