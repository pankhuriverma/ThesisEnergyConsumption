import pyRAPL

pyRAPL.setup()

@pyRAPL.measureit(number=100)
def foo():
   print("hi")

foo()