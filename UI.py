## 2048 game user interface (UI), using tkinter
from tkinter import *
from board import Board
from AI import AI

import string
import sys


class UI:
    def __init__(self):
        self.GameBoard = Board(4)  # default board size is 4

    def init(self, data):
        # board size settings
        data.size = self.GameBoard.size
        data.margin = 25
        data.titlePlace = 150
        data.cellSize = (data.width - data.margin * 2) / data.size
        data.board = self.GameBoard.board
        data.finishAdding = False
        data.newTileIndex = None, None  # index of the new adding tile

        # start page modes
        data.startPage = True
        data.playerMode = False
        data.AImode = False

        # size, level selection page modes
        data.customizeSizeMode = False
        data.selectingBoardSize = False

        data.levelSelectionMode = False  # only applicable for AI mode

        # game page modes
        data.inGame = False

        # game state settings (Game Over, Game Paused)
        data.reach2048 = False  # reach 2048 does not necessarily means game over here
        data.continuePlaying = False  # continue the game after reaching 2048
        data.cannotMove = False  # do not have any valid move means game over
        data.paused = False

        # time settings
        data.timerDelay = 1000
        data.timeCounter = 0

        # color settings
        data.colors = {
            0: "#D7CCC8",
            2: "#FFFDE7",
            4: "#FBC02D",
            8: "#F9A825",
            16: "#FFA000",
            32: "#F57F17",
            64: "#F44336",
            128: "#FFEB3B",
            256: "#FFD600",
            512: "#FDD835",
            1024: "#FFFF8D",
            2048: "#FFFF00",
            4096: "#616161",
            8192: "#424242",
            16384: "#212121",
            32768: "#03A9F4",
            65536: "#039BE5",
            131072: "#0288D1",
        }

        # AI settings
        data.AIstep = 0
        data.AI = None

    def resetGameBoard(self, data):
        self.GameBoard = Board(data.size)
        data.board = self.GameBoard.board
        data.cellSize = (data.width - data.margin * 2) / data.size
        data.timeCounter = 0
        data.reach2048 = False
        data.continuePlaying = False
        data.cannotMove = False

    def mousePressed(self, event, data):
        # three buttons in the start page
        if data.startPage:
            # enter the player mode
            if data.width // 4 <= event.x <= data.width * (
                3 / 4
            ) and data.height // 2 <= event.y <= data.height * (3 / 5):
                data.playerMode = True
                data.customizeSizeMode = True
                data.startPage = False

            # enter the AI mode
            elif data.width // 4 <= event.x <= data.width * (3 / 4) and data.height * (
                3 / 5 + 1 / 30
            ) <= event.y <= data.height * (7 / 10 + 1 / 30):
                data.AImode = True
                data.customizeSizeMode = True
                data.startPage = False

            # exit button
            elif data.width * (5 / 6) <= event.x <= data.width * (
                29 / 30
            ) and data.height * (9 / 10) <= event.y <= data.height * (29 / 30):
                sys.exit()

        # customize size page
        elif data.customizeSizeMode:
            # press the empty "size button" to enter the board size
            if data.width // 2 <= event.x <= data.width * (
                3 / 4
            ) and data.height // 2 <= event.y <= data.height * (3 / 5):
                data.selectingBoardSize = True

            # press the finish button to enter the game state
            elif data.width * (3 / 8) <= event.x <= data.width * (
                5 / 8
            ) and data.height * (1 / 4 + 1 / 15) <= event.y <= data.height * (
                7 / 20 + 1 / 15
            ):
                self.resetGameBoard(data)
                if data.AImode:
                    data.levelSelectionMode = True
                else:
                    data.inGame = True
                data.customizeSizeMode = False

        # level selection mode (only applicable for AI mode)
        elif data.levelSelectionMode:
            assert data.AImode and not data.playerMode

            # novice
            if data.width // 4 <= event.x <= data.width * (
                3 / 4
            ) and data.height // 2 <= event.y <= data.height * (3 / 5):
                data.AI = AI(self.GameBoard, 0)
                data.levelSelectionMode = False
                data.inGame = True

            # competent
            elif data.width // 4 <= event.x <= data.width * (3 / 4) and data.height * (
                3 / 5 + 1 / 30
            ) <= event.y <= data.height * (7 / 10 + 1 / 30):
                data.AI = AI(self.GameBoard, 1)
                data.levelSelectionMode = False
                data.inGame = True

            # expert
            elif data.width // 4 <= event.x <= data.width * (3 / 4) and data.height * (
                7 / 10 + 1 / 15
            ) <= event.y <= data.height * (4 / 5 + 1 / 15):
                data.AI = AI(self.GameBoard, 2)
                data.levelSelectionMode = False
                data.inGame = True

    def keyPressed(self, event, data):
        # customize size mode
        if data.customizeSizeMode:
            if data.selectingBoardSize and event.keysym in string.digits:
                if int(event.keysym) in range(4, 10):
                    data.size = int(event.keysym)
                elif int(event.keysym) == 1:  # enter 1 to represent size = 10
                    data.size = 10
                data.selectingBoardSize = False

        # in game mode
        if data.inGame:
            # pause the game (can be retrieved)
            if event.keysym == "p":
                data.paused = not data.paused

            # player cannot move in the AI mode
            if data.AImode:
                pass
            else:
                # making moves based on the user's pressing keys
                if not self.GameBoard.gameOver():
                    if self.GameBoard.reaches2048():
                        data.reach2048 = True
                        if not data.continuePlaying:
                            data.inGame = False

                    canMove = False
                    if data.paused:
                        print(
                            "the game is paused now! you cannot move until the game is unpaused"
                        )
                    else:
                        direction = event.keysym
                        if direction == "Up":
                            canMove = self.GameBoard.moveUp()
                        elif direction == "Down":
                            canMove = self.GameBoard.moveDown()
                        elif direction == "Left":
                            canMove = self.GameBoard.moveLeft()
                        elif direction == "Right":
                            canMove = self.GameBoard.moveRight()

                        # add a new number after each legal move
                        if canMove:
                            (
                                data.newTileIndex,
                                data.newTileNum,
                            ) = self.GameBoard.addNewTile()
                            print("data.newTileIndex: ", data.newTileIndex)
                        else:
                            print("cannot move in this direction")
                        self.GameBoard.printBoard()
                    data.board = self.GameBoard.board
                    data.finishAdding = False  # set to False for next move

                else:
                    data.cannotMove = True
                    data.inGame = False
                    print("Game Over!")
        else:
            # restart the game only when the current game is over
            if event.keysym == "r":
                data.inGame = True
                self.resetGameBoard(data)
                if data.AImode:
                    self.init(data)  # for AI mode, directly return to the start page

            # press c to continue playing the game
            elif event.keysym == "c" and data.reach2048 and not data.cannotMove:
                data.inGame = True
                data.continuePlaying = True

    ## Graphic drawing functions below
    # home page
    def drawStartPage(self, canvas, data):
        # start page
        canvas.create_rectangle(0, 0, data.width, data.height, fill="#ECEFF1")
        canvas.create_text(
            data.width // 2,
            data.height // 4,
            text="2048",
            fill="#FDD835",
            font="Oswald 90 bold",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (3 / 8),
            text="by Yilun Wu",
            fill="light coral",
            font="Caladea 40",
        )

        # player mode button
        canvas.create_rectangle(
            data.width // 4,
            data.height // 2,
            data.width * (3 / 4),
            data.height * (3 / 5),
            fill="lemon chiffon",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (11 / 20),
            text="Player",
            font="Arial 40 bold",
        )

        # AI mode button
        canvas.create_rectangle(
            data.width // 4,
            data.height * (3 / 5 + 1 / 30),
            data.width * (3 / 4),
            data.height * (7 / 10 + 1 / 30),
            fill="lemon chiffon",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (13 / 20 + 1 / 30),
            text="AI",
            font="Arial 40 bold",
        )

        # settings button
        canvas.create_rectangle(
            data.width // 4,
            data.height * (7 / 10 + 1 / 15),
            data.width * (3 / 4),
            data.height * (4 / 5 + 1 / 15),
            fill="lemon chiffon",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (3 / 4 + 1 / 15),
            text="Settings",
            font="Arial 40 bold",
        )

        # quit button
        canvas.create_rectangle(
            data.width * (5 / 6),
            data.height * (9 / 10),
            data.width * (29 / 30),
            data.height * (29 / 30),
            fill="light grey",
        )
        canvas.create_text(
            data.width * (9 / 10),
            data.height * (14 / 15),
            text="Quit",
            font="Arial 30 bold",
        )

    # customize size page
    def drawCustomizeSizePage(self, canvas, data):
        canvas.create_rectangle(0, 0, data.width, data.height, fill="cyan")
        canvas.create_text(
            data.width // 2,
            data.height // 4,
            text="Select Your Size Here!",
            font="Arial 50 bold",
            fill="purple",
        )

        # sizes (4-10)
        canvas.create_text(
            data.width // 4,
            data.height * (11 / 20),
            text="Board Size(Width):",
            font="Arial 30",
        )
        if data.selectingBoardSize:
            canvas.create_rectangle(
                data.width // 2,
                data.height // 2,
                data.width * (3 / 4),
                data.height * (3 / 5),
                fill="light goldenrod",
            )
        else:
            canvas.create_rectangle(
                data.width // 2,
                data.height // 2,
                data.width * (3 / 4),
                data.height * (3 / 5),
                fill="lemon chiffon",
            )
        canvas.create_text(
            data.width * (5 / 8),
            data.height * (11 / 20),
            text=data.size,
            font="Arial 30",
        )
        canvas.create_text(
            data.width * (7 / 8),
            data.height * (11 / 20),
            text="(4-10)",
            font="Arial 30",
        )

        # finish button
        canvas.create_rectangle(
            data.width * (3 / 8),
            data.height * (1 / 4 + 1 / 15),
            data.width * (5 / 8),
            data.height * (7 / 20 + 1 / 15),
            fill="lemon chiffon",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (3 / 10 + 1 / 15),
            text="Finish!",
            font="Arial 35",
        )

    # AI mode level selection page
    def drawLevelSelectionPage(self, canvas, data):
        canvas.create_rectangle(0, 0, data.width, data.height, fill="cyan")
        canvas.create_text(
            data.width // 2,
            data.height // 4,
            text="Select a Level!",
            font="Arial 55 bold",
            fill="purple",
        )

        # novice mode
        canvas.create_rectangle(
            data.width // 4,
            data.height // 2,
            data.width * (3 / 4),
            data.height * (3 / 5),
            fill="lemon chiffon",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (11 / 20),
            text="Novice",
            font="Arial 35 bold",
        )

        # competent mode
        canvas.create_rectangle(
            data.width // 4,
            data.height * (3 / 5 + 1 / 30),
            data.width * (3 / 4),
            data.height * (7 / 10 + 1 / 30),
            fill="lemon chiffon",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (13 / 20 + 1 / 30),
            text="Competent",
            font="Arial 35 bold",
        )

        # expert mode
        canvas.create_rectangle(
            data.width // 4,
            data.height * (7 / 10 + 1 / 15),
            data.width * (3 / 4),
            data.height * (4 / 5 + 1 / 15),
            fill="lemon chiffon",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (3 / 4 + 1 / 15),
            text="Expert",
            font="Arial 35 bold",
        )

    # game page
    def drawCell(self, canvas, data, row, col):
        # draw every cell
        currNum = data.board[row][col]
        cellBoundsWidth = 2.5
        if (
            row,
            col,
        ) == data.newTileIndex and not data.finishAdding:  # new added tile with contrasting color
            canvas.create_rectangle(
                data.margin + data.cellSize * col,
                data.margin + data.titlePlace + data.cellSize * row,
                data.margin + data.cellSize * (col + 1),
                data.margin + data.titlePlace + data.cellSize * (row + 1),
                fill="cyan",
                width=cellBoundsWidth,
            )
            data.finishAdding = True
            data.newTileIndex = None, None
        else:
            canvas.create_rectangle(
                data.margin + data.cellSize * col,
                data.margin + data.titlePlace + data.cellSize * row,
                data.margin + data.cellSize * (col + 1),
                data.margin + data.titlePlace + data.cellSize * (row + 1),
                fill=data.colors[currNum],
                width=cellBoundsWidth,
            )

    def drawBoard(self, canvas, data):
        # draw the board by filling every cells(using draw cells)
        for row in range(data.size):
            for col in range(data.size):
                self.drawCell(canvas, data, row, col)
                if data.board[row][col]:
                    textSize = 180 // data.size
                    canvas.create_text(
                        data.margin + data.cellSize / 2 + col * data.cellSize,
                        data.titlePlace
                        + data.margin
                        + data.cellSize / 2
                        + row * data.cellSize,
                        text=data.board[row][col],
                        font=("Arial", textSize),
                        fill="black",
                    )

    def drawGamePage(self, canvas, data):
        canvas.create_rectangle(0, 0, data.width, data.height, fill="#EFEBE9")
        canvas.create_text(
            data.width // 4,
            data.titlePlace // 2,
            text="2048",
            font="Arial 60 bold",
            fill="#795548",
        )
        if data.reach2048:
            canvas.create_text(
                data.width // 4,
                data.titlePlace * (3 / 4),
                text="Reached 2048",
                font="TimesNewRoman 30 bold",
                fill="red",
            )

        if data.AImode:  # AI mode
            canvas.create_text(
                data.width // 2,
                data.titlePlace // 2,
                text="Step:" + str(data.AIstep),
                font="Arial 23 bold",
                fill="purple",
            )
        else:  # player mode
            canvas.create_text(
                data.width // 2,
                data.titlePlace // 2,
                text="Time:" + str(data.timeCounter),
                font="Arial 23 bold",
                fill="purple",
            )
        canvas.create_text(
            data.width * (3 / 4),
            data.titlePlace // 2,
            text="Score:" + str(self.GameBoard.score),
            font="Arial 23 bold",
            fill="purple",
        )
        self.drawBoard(canvas, data)

        # Game paused
        if data.paused:
            canvas.create_rectangle(
                0, data.height // 3, data.width, data.height * (2 / 3), fill="gold"
            )
            canvas.create_text(
                data.width // 2,
                data.height // 2,
                text="Game Paused!",
                font="TimesNewRoman 35 bold",
                fill="red",
            )

    def drawReach2048Page(self, canvas, data):
        # reach 2048
        canvas.create_rectangle(0, 0, data.width, data.height, fill="#EEEBE9")
        canvas.create_text(
            data.width // 2,
            data.height // 4,
            text="Game Score: " + str(self.GameBoard.score),
            font="Arial 30 bold",
            fill="purple",
        )
        canvas.create_text(
            data.width // 2,
            data.height // 2,
            text="Congratulations!",
            font="Arial 50 bold",
            fill="red",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (3 / 5),
            text="you get 2048 and WIN!",
            font="Arial 50 bold",
            fill="red",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (3 / 4),
            text="press 'r' to restart",
            font="Arial 30 bold",
            fill="purple",
        )
        # game over after reaching 2048
        if not data.cannotMove:
            canvas.create_text(
                data.width // 2,
                data.height * (4 / 5),
                text="press 'c' to continue",
                font="Arial 30 bold",
                fill="purple",
            )

    # game over page
    def drawGameOverPage(self, canvas, data):
        canvas.create_rectangle(0, 0, data.width, data.height, fill="#EEEBE9")
        canvas.create_text(
            data.width // 2,
            data.height // 2,
            text="You LOSE",
            font="Arial 50 bold",
            fill="black",
        )
        canvas.create_text(
            data.width // 2,
            data.height // 4,
            text="Game Score: " + str(self.GameBoard.score),
            font="Arial 30 bold",
            fill="purple",
        )
        canvas.create_text(
            data.width // 2,
            data.height * (3 / 4),
            text="press 'r' to restart",
            font="Arial 30 bold",
            fill="purple",
        )

    def redrawAll(self, canvas, data):
        # start page
        if data.startPage:
            self.drawStartPage(canvas, data)

        # customize size page
        elif data.customizeSizeMode:
            self.drawCustomizeSizePage(canvas, data)

        # AI mode level selection page
        elif data.levelSelectionMode:
            assert data.AImode and not data.playerMode and not data.customizeSizeMode
            self.drawLevelSelectionPage(canvas, data)

        # in game page
        elif data.inGame:
            self.drawGamePage(canvas, data)

        # reach 2048 page
        elif data.reach2048:
            self.drawReach2048Page(canvas, data)

        # game over page
        elif data.cannotMove:
            self.drawGameOverPage(canvas, data)

    def AImove(self, data):
        print("AI playing")
        print("step %d:" % data.AIstep)
        data.AI.nextMove()
        # update the board after each AI's move
        self.GameBoard.board = data.AI.GameBoard.board
        self.GameBoard.printBoard()
        data.board = data.AI.GameBoard.board
        data.AIstep += 1

    def timerFired(self, data):
        if data.inGame and not (data.reach2048 or data.cannotMove) and not data.paused:
            data.timeCounter += 1

            if data.AImode:
                data.timerDelay = 200
                if not self.GameBoard.gameOver():
                    # move twice in each timer delay period
                    self.AImove(data)
                else:
                    data.inGame = False
                    if self.GameBoard.reaches2048():
                        data.reach2048 = True
                    else:
                        data.cannotMove = True

    def runGame(self, width, height):  # tkinter starter code
        def redrawAllWrapper(canvas, data):
            canvas.delete(ALL)
            canvas.create_rectangle(
                0, 0, data.width, data.height, fill="white", width=0
            )
            self.redrawAll(canvas, data)
            canvas.update()

        def mousePressedWrapper(event, canvas, data):
            self.mousePressed(event, data)
            redrawAllWrapper(canvas, data)

        def keyPressedWrapper(event, canvas, data):
            self.keyPressed(event, data)
            redrawAllWrapper(canvas, data)

        def timerFiredWrapper(canvas, data):
            self.timerFired(data)
            redrawAllWrapper(canvas, data)
            # pause, then call timerFired again
            canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)

        # Set up data and call init
        class Struct(object):
            pass

        data = Struct()
        data.width = width
        data.height = height
        data.timerDelay = 500  # milliseconds
        root = Tk()
        root.title("2048")
        root.resizable(width=False, height=False)  # prevents resizing window
        self.init(data)
        # create the root and the canvas
        canvas = Canvas(root, width=data.width, height=data.height)
        canvas.configure(bd=0, highlightthickness=0)
        canvas.pack()
        # set up events
        root.bind("<Button-1>", lambda event: mousePressedWrapper(event, canvas, data))
        root.bind("<Key>", lambda event: keyPressedWrapper(event, canvas, data))
        timerFiredWrapper(canvas, data)
        # and launch the app
        root.mainloop()  # blocks until window is closed
