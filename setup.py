import subprocess
import sys


def loop(question, action, neg_message):
    success = 0
    try:
        while not success:
            res = input(f"---> {question} [y/n]\n")
            if "y" in res or "Y" in res:
                action()
                success = 1
            elif "n" in res or "N" in res:
                print(f"---> {neg_message}\n")
                success = 1
    except Exception as e:
        print(f"ERROR - loop - {e}")


def main():
    loop(
        "Update dependencies?",
        lambda: subprocess.check_call(
            [sys.executable, *"-m pip install -r requirements.txt".split(" ")]
        ),
        "Skipping dependencies...",
    )
    print("Finished")


if __name__ == '__main__':
    main()
