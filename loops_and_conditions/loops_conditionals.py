### This shows the logic of loops and conditionals

### Example of a python program that find numbers that are devisable by 7 and are multiples of 5, between 1500 and 2700.
nl=[]
for x in range(1500, 2701):
    if (x%7==0) and (x%5==0):
        nl.append(str(x))
print (','.join(nl))

### Write a Python program to convert temperatures to and from celsius, fahrenheit. Go to the editor
### [ Formula : c/5 = f-32/9 [ where c = temperature in celsius and f = temperature in fahrenheit ]
temp = input("Input the  temperature you like to convert? (e.g., 45F, 102C etc.) : ")
degree = int(temp[:-1])
i_convention = temp[-1]

if i_convention.upper() == "C":
  result = int(round((9 * degree) / 5 + 32))
  o_convention = "Fahrenheit"
elif i_convention.upper() == "F":
  result = int(round((degree - 32) * 5 / 9))
  o_convention = "Celsius"
else:
  print("Input proper convention.")
  quit()
print("The temperature in", o_convention, "is", result, "degrees.")


### User is prompted to enter a guess. If the user guesses wrong then the prompt appears again until the guess is
### correct, on successful guess, user will get a "Well guessed!" message, and the program will exit.
import random
target_num, guess_num = random.randint(1, 10), 0
while target_num != guess_num:
    guess_num = int(input('Guess a number between 1 and 10 until you get it right : '))
print('Well guessed!')
