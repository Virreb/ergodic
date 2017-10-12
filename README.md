# ergodic
Python3.6 is used in this project.

## Setup
To install modules, you need the python package manager pip (sudo apt-get install pip). 

### Virtual environment
It is recommended to use a virtual environment to install packages in, so they don't get installed globally. There are many tools for this but the most used low-lever is called virtualenv.

> pip install virtualenv

Create a virtual environment for a project:

> cd my_project_folder && virtualenv env -p python3.6

This creates a copy of Python3.6 in whichever directory you ran the command in, placing it in a folder named 'env'. To begin using the virtual environment, it needs to be activated:

> source env/bin/activate

This will now show in the terminal prompt. To deactivate the venv, just use the command

> deactivate

In Pycharm, you can also choose the virtualenv to be the standard python interpreter.

### Packages
The easiest way to install the required packages is to read the requirements.txt file in the project root. Don't forget to activate your virtual environment first!

> pip install -r requirements.txt

If some package is added later on, you can simply create a new requirements.txt file with

> pip freeze > requirements.txt
