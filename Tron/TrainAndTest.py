import TronModel
import tensorflow as tf
import random

def trainOneGame(startState):
    



def main():

    tm = TronModel()
    mapList = []
    for i in range(1000):
        trainOneGame(random.choice(mapList))
    tm.save("trainedModel")

if __name__ == "__main__":
    main()
