import os

from colorama import Fore

from sip import load_chore, make_chore_from_file, save_chore
from sip import dance_along
from sip import cosine_similarity


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
        trainee = make_chore_from_file(
            f"trainee_{choregraphies[choice]}",
            "./tmp.mp4",
            load_message="We are computing your score ... ",
        )

        score, proportion = cosine_similarity(chore, trainee)
        score *= 100

        trainee.score = score
        save_chore(trainee, "./trainee_chores/")
        os.remove("./tmp.mp4")

        print(
            "Your final score is "
            + Fore.RED
            + f"{int(score)}"
            + Fore.RESET
            + f" with joint visibility of {proportion}"
        )

        while True:
            choice = input("Train again ? [y/n] : ")
            if choice in ["y", "n"]:
                break

        if choice == "y":
            continue
        else:
            break

    print("Thanks for playing !")
