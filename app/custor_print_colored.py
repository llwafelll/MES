from termcolor import colored

def decorator(*args, **kwargs):
    def wrapper(text):
        print(colored(text, *args, **kwargs))
    
    return wrapper

print_H1 = decorator("white", "on_red", attrs=("bold", ))
print_H2 = decorator("red", attrs=("bold", "underline"))
print_H3 = decorator("red")

if __name__ == "__main__":
    print_H1("This is H1")
    print_H2("This is H2")
    print_H3("This is H3")