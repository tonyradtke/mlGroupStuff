# mlGroupStuff

drugs.csv:
a slightly modified version of that data we put together (i just got rid of the 'CL' character in favor of just the numbers)

drugs.py:
uses a simple nn to do binary classification from the input stats to one of the drug columns. 
note: the drug columns have several classes (CL1, CL2 etc), but i just reduced them into either 0 or 1 for now

test accuracy is about 60% right now. 
I havent spent any time at all yet tweaking the NN architecture or hyperparamters to see if we can get a better result.
Most useful thing to tweak are the number of epochs, the learning rate, number of layers/nodes, and the optimization method.

not sure yet what the end thing we want is, but this should be a decent place to start.


 :)

