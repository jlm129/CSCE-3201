import numpy as np
import random
l = 10
w = 10
my_array = np.zeros((w,l))
cor_x = random.randint(0, w-1)
cor_y = random.randint(0, l-1)
my_array[cor_x,cor_y] = 1
correct = False
guesses = 0
max_attempts = 10

while correct is False:
    my_guess = input("Guess the row position of the treasure: ")
    if(int(my_guess) < cor_y):
        guesses = guesses + 1
        print(f"You guessed too low for the row position, try again, {guesses} used!")
    elif(int(my_guess) > cor_y):
        guesses = guesses + 1
        print(f"You guessed too high for the row position, try again, {guesses} used!")
    else:
        print(f"You guessed correctly, {guesses} attempt used, now try to guess the column position!")
        correct = True
    if(guesses> max_attempts):
        print(f"Too many incorrect guesses the location was {cor_x, cor_y}")

correct = False

while correct is False:

    my_guess = input("Guess the col position of the treasure: ")
    if(int(my_guess) < cor_x):
        guesses = guesses + 1
        print(f"You guessed too low for the col position, try again, {guesses} used!")
    elif(int(my_guess) > cor_x):
        guesses = guesses + 1
        print(f"You guessed too high for the col position, try again, {guesses} used!")
    else:
        print(f"You guessed correctly, {guesses} attempt used")
        correct = True
    if(guesses>max_attempts):
        print(f"Too many incorrect guesses the location was {cor_x, cor_y}")
