# From https://www.youtube.com/watch?v=VqChpNNYZ8Q

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def activate(x):
    return 0 if x < 0.5 else 1

def decide(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]

    weight1 = np.array([w11, w12])
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, x)
    print("Hidden layer input: " + str(sum_hidden))

    out_hidden = np.array([activate(x) for x in sum_hidden])
    print("Hidden layer output: " + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = activate(sum_end)
    print("Output: " + str(y))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    if decide(1, 0, 1) == 1:
        print("Yes")
    else:
        print("No")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
