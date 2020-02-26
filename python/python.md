```
# get path for current working directory
print(os.getcwd())

# get path for current file
print(os.path.realpath(__file__))

# get path for the directory of current file
print(os.path.dirname(os.path.realpath(__file__)))
```
