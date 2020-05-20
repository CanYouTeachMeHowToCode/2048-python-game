## 2048 game user interface (UI), using tkinter
from tkinter import *
from board import Board
#from AI import AI
import copy

class UI(object):
    def __init__(self, size):
        self.size = size
        self.GameBoard = Board(self.size)

    def init(self, data):
        data.size = self.size
        data.margin = 25
        data.titlePlace = 150
        data.cellSize = (data.width - data.margin * 2) / data.size
        data.GameOver = False
        data.timeCounter = 0
        data.timerDelay = 1000
        data.colors = {0 : "#D7CCC8",
                       2 : "#FFFDE7", 
                       4 : "#FBC02D",
                       8 : "#F9A825",
                       16 : "#FFA000",
                       32 : "#F57F17",
                       64 : "#F44336",
                       128 : "#FFEB3B",
                       256 : "#FFD600",
                       512 : "#FDD835",
                       1024 : "#FBC02D",
                       2048 : "#FFFF00",
                       4096 : "#616161",
                       8192 : "#424242",
                       16384 : "#212121",
                       32768 : "#03A9F4",
                       65536 : "#039BE5",
                       131072 : "#0288D1"}

    def mousePressed(self, event, data):
        pass

    def keyPressed(self, event, data):
        if not self.GameBoard.GameOver():
            canMove = False
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
            if canMove: self.GameBoard.addNewTile() 
            else: print("cannot move in this direction") 
            self.GameBoard.printBoard()
        else:
            data.GameOver = True
            print("Game Over!")

    def drawCell(self, canvas, data, row, col):
        #draw every cell
        currNum = self.GameBoard.board[row][col]
        cellBoundsWidth = 2.5
        canvas.create_rectangle(data.margin + data.cellSize*col, data.margin + \
        data.titlePlace + data.cellSize*row, data.margin + data.cellSize*(col+1), \
        data.margin + data.titlePlace + data.cellSize*(row+1), \
        fill = data.colors[currNum], width = cellBoundsWidth)

    def drawBoard(self, canvas, data):
        #draw the board by filling every cells(using draw cells)
        for row in range(data.size):
            for col in range(data.size):
                self.drawCell(canvas, data, row, col)
                if self.GameBoard.board[row][col]:
                    canvas.create_text(data.margin + data.cellSize/2 + 
                        col*data.cellSize, data.titlePlace + data.margin + 
                        data.cellSize/2 + row*data.cellSize, 
                        text = self.GameBoard.board[row][col], \
                        font = "Arial 45", fill = "black") 

    def redrawAll(self, canvas, data):
        canvas.create_rectangle(0, 0, data.width, data.height, fill = "#EFEBE9")
        canvas.create_text(data.width / 4, data.titlePlace / 2, 
                            text = "2048", \
                            font = "Arial 60 bold", fill = "#795548")
        canvas.create_text((data.width / 2), data.titlePlace / 2, 
                            text = "Time:" + str(data.timeCounter) ,\
                            font = "Arial 23 bold", fill = "purple")
        canvas.create_text((data.width * 0.75), data.titlePlace / 2, 
                            text = "Score:" + str(self.GameBoard.score) ,\
                            font = "Arial 23 bold", fill = "purple")
        self.drawBoard(canvas, data)

    def timerFired(self, data):
        if not data.GameOver:
            data.timeCounter += 1

    def runGame(self, width, height): # tkinter starter code
        def redrawAllWrapper(canvas, data):
            canvas.delete(ALL)
            canvas.create_rectangle(0, 0, data.width, data.height,
                                    fill='white', width=0)
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
        class Struct(object): pass
        data = Struct()
        data.width = width
        data.height = height
        data.timerDelay = 500 # milliseconds
        root = Tk()
        root.title("2048")
        root.resizable(width=False, height=False) # prevents resizing window
        self.init(data)
        # create the root and the canvas
        canvas = Canvas(root, width=data.width, height=data.height)
        canvas.configure(bd=0, highlightthickness=0)
        canvas.pack()
        # set up events
        root.bind("<Button-1>", lambda event:
                                mousePressedWrapper(event, canvas, data))
        root.bind("<Key>", lambda event:
                                keyPressedWrapper(event, canvas, data))
        timerFiredWrapper(canvas, data)
        # and launch the app
        root.mainloop()  # blocks until window is closed