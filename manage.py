# manage

import sys
import os

argv = sys.argv

name = argv[0]
command = argv[1]

def run_command(path):
    if sys.version_info[0] < 3:
        print("Python version must be 3\ntry python3 manage.py {argv}")
        return
    os.system("py " + path)


def main():
    if command in ["train", "evl","server"]:
        if command == "train":
            run_command("src/train_md.py")

        elif command == "evl":
            run_command("src/utils.py")

        else:
            run_command("src/index.py")
        
        return "\nExit."

    return "command '" + command + "' not found!"


print(main())
