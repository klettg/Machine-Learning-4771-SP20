References/Dependencies:

shutil copy2: used for copying/creating new directories to help split up test/train data
nltk: used for stemming words
string, os, math, csv
all used for auxilary functions and for writing output

How to run code: 
I have three pycharm projects, one for each type of classifier.
Inside each project, there is a variable called DocumentDirectory with the path location to the folder containing the email data. Currently this is hardcoded to where I read the documents from on my computer. 

Update this variable with the correct path and then execute the main.py in each project to run each classfiier on the data in the documentDirectory path. This will automatically split the data into test/training sets. 
