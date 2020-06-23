### Lists
thislist = ["apple", "banana", "cherry"]
print(thislist)

## Print the second item of the list:
thislist = ["apple", "banana", "cherry"]
print(thislist[1])


## Negative indexing means beginning from the end, -1 refers to the last item, -2 refers to the second last item etc.
thislist = ["apple", "banana", "cherry"]
print(thislist[-1])

## To change the value of a specific item, refer to the index number:
thislist = ["apple", "banana", "cherry"]
thislist[1] = "blackcurrant"
print(thislist)

## Print all items in the list, one by one:
thislist = ["apple", "banana", "cherry"]
for x in thislist:
  print(x)

## Empty list
thatlist = []
for i in range(1000):
    thatlist = thatlist.append(i)

## length of the list:
print(len(thatlist))
print(len(thislist))


### Dictionaries

## A dictionary is a collection which is unordered, changeable and indexed. In Python dictionaries are written with curly brackets, and they have keys and values.

thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)

## Get the entry of a specific value in a dictionary

x = thisdict["model"]
# The .get command gives you the same result.
x = thisdict.get("model")

## Change the value for the 'year' to 2018

thisdict["year"] = 2018

## Print all key names in the dictionary, one by one:

for x in thisdict:
  print(x)

## Print all values in the dictionary, one by one:

for x in thisdict:
  print(thisdict[x])

## You can also use the values() method to return values of a dictionary:

for x in thisdict.values():
  print(x)

## Loop through both keys and values, by using the items() method:

for x, y in thisdict.items():
  print(x, y)

## Check if "model" is present in the dictionary:

if "model" in thisdict:
  print("Yes, 'model' is one of the keys in the thisdict dictionary")
else:
    print("No, there is not a key in thisdictionary with the name 'model'")

## Print the number of items in the dictionary:

print(len(thisdict))

## Adding an item to the dictionary is done by using a new index key and assigning a value to it:

thisdict["color"] = "red"
print(thisdict)

## Removing ites from a dictionary:
## The pop() method removes the item with the specified key name:

thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict.pop("model")
print(thisdict)
#The popitem() method removes the last inserted item (in versions before 3.7, a random item is removed instead):

## The clear() method empties the dictionary:

thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict.clear()
print(thisdict)

## Make a copy of a dictionary with the copy() method:

thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
mydict = thisdict.copy()
print(mydict)

## A dictionary can also contain many dictionaries, this is called nested dictionaries.
myfamily = {
  "child1" : {
    "name" : "Emil",
    "year" : 2004
  },
  "child2" : {
    "name" : "Tobias",
    "year" : 2007
  },
  "child3" : {
    "name" : "Linus",
    "year" : 2011
  }
}

### Create a dictionary out of my family, which just have the entries for names.