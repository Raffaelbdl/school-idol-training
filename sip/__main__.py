import os

from colorama import Fore

from sip.chore_tools.chore_creator import (
    load_chore,
    make_chore_from_file,
    save_chore,
)
from sip.chore_training.trainee import dance_along
from sip.chore_training.scoring import get_score

if __name__ == "__main__":
    print(
        "Welcome to " + Fore.MAGENTA + "'The School Idol Project'" + Fore.RESET + "!!"
    )
    print("Are you ready to become an " + Fore.MAGENTA + "IDOL" + Fore.RESET + "??")

    while True:
        choregraphies = os.listdir("./choregraphies")
        print("\nAvailable choregraphies are :")
        for i, chore_name in enumerate(choregraphies):
            print(f"{i} - {chore_name}")
        while True:
            choice = int(input("What do you want to train on ? : "))
            if choice in range(len(choregraphies)):
                break

        chore = load_chore(os.path.join("./choregraphies", choregraphies[choice]))

        print("Please set up your camera")
        print("Press 'Q' to start")

        dance_along(chore)
        trainee = make_chore_from_file(f"trainee_{choregraphies[choice]}", "./tmp.mp4")
        save_chore(trainee, "./trainee_chore/")
        os.remove("./tmp.mp4")

        score = get_score(chore, trainee, "fast_dtw")
        print(f"Your final score is {score}")

        while True:
            choice = input("Train again ? [y/n] : ")
            if choice in ["y", "n"]:
                break

        if choice == "y":
            continue
        else:
            break

    print("Thanks for playing !")
