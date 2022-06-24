import os

from colorama import Fore

from sip import load_chore, launch_game

choregraphies_dir_path = "./choregraphies"
difficulty = 0  # 0 easy, 1 medium, 2 hard


if __name__ == "__main__":
    print(
        "\nWelcome to " + Fore.MAGENTA + "'The School Idol Project'" + Fore.RESET + "!!"
    )
    print("Are you ready to become an " + Fore.MAGENTA + "IDOL" + Fore.RESET + "??")

    while True:
        try:
            choregraphies = os.listdir(choregraphies_dir_path)
        except FileNotFoundError:
            print(f"\ndirectory '{choregraphies_dir_path}' does not exist")
            print("Please specify another directory in __main__.py")
            break

        if len(choregraphies) == 0:
            print(Fore.RED + "\nYou don't have any choregraphy !" + Fore.RESET)
            print("Please find how to make one in the README file")
            print("Or at https://github.com/Raffaelbdl/school-idol-training")
            break

        print("\nAvailable choregraphies are :")
        for i, chore_name in enumerate(choregraphies):
            print(f"{i} - {chore_name}")
        while True:
            choice = int(input("What do you want to train on ? : "))
            if choice in range(len(choregraphies)):
                break

        chore = load_chore(os.path.join(choregraphies_dir_path, choregraphies[choice]))

        print("Please set up your camera")
        print("Press 'Q' to start")

        score, visibility = launch_game(chore, difficulty)

        print(
            "Your final score is "
            + Fore.RED
            + f"{int(score)}"
            + Fore.RESET
            + f" with joint visibility of {visibility:.2f}"
        )

        while True:
            choice = input("Train again ? [y/n] : ")
            if choice in ["y", "n"]:
                break

        if choice == "y":
            continue
        else:
            break

    print("\nThanks for playing !\n")
