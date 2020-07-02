### Example of classes and objects in python

# A general object of the type person is created
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

# Note: The __init__() method is commonly known as an initializer method because it initializes the object's data attributes.
#       An object of a class usually initialized by implementing the __init__() method. When an object is created, Python
#       first creates an empty object and then calls the __init__() method for that new object.

class Ball:
  def __init__(self, weight, diameter, colour):
    self.weight = weight
    self.diameter = diameter
    self.colour = colour

  def details(self):
    print("Weight of the ball in grams", self.weight)
    print("Diameter of the ball in cm", self.diameter)
    print("Colour of the ball", self.colour)


p1 = Person("John", 36)

print(p1.name)
print(p1.age)

print(type(p1))

bowling = Ball(2500, 25, "Black")
cricket = Ball(250, 10, "Red")

print(bowling.colour, cricket.colour)
print(type(bowling))