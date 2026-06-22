import sys  # సిస్టమ్ ఎగ్జిట్ (sys.exit) కోసం
from colorama import Fore
# 🎯 మీ లోకల్ ఫోల్డర్ లోని query.py నుండి పర్ఫెక్ట్ గా లింక్ అవుతుంది
from query5 import query 


def start():
    instructions = (
        """Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"""
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        sys.exit()  # పాత exit() కి బదులు sys.exit() వాడటం ప్రొడక్షన్ స్టాండర్డ్
    else:
        print("Invalid choice")
        start()


def ask():
    while True:
        user_input = input("Q: ")
        
        # Exit లాజిక్ ని పక్కాగా అప్‌గ్రేడ్ చేసాము
        if user_input.lower() == "x":
            start()
            break  # 🎯 పాత లూప్ బ్యాక్‌గ్రౌండ్ లో రన్ అవ్వకుండా క్లోజ్ అవ్వడానికి break వాడాము
        
        # ఒకవేళ యూజర్ ఏమీ టైప్ చేయకుండా ఎంటర్ నొక్కితే స్కిప్ చేయడానికి
        if not user_input.strip():
            continue

        try:
            # query.py లోని అప్‌డేటెడ్ LCEL RAG ప్రాసెస్ ని రన్ చేస్తుంది
            response = query(user_input)

            # మీ రిక్వైర్మెంట్ ప్రకారం response["answer"] ని ప్రింట్ చేస్తుంది
            print(Fore.BLUE + "A: " + response["answer"] + Fore.RESET)
            print(Fore.WHITE + 
                  "\n-------------------------------------------------")
        except Exception as e:
            # రన్‌టైమ్ లో ఏదైనా నెట్‌వర్క్ సమస్య వస్తే క్రాష్ అవ్వకుండా ఎర్రర్ హ్యాండ్లింగ్
            print(Fore.RED + f"Error: {e}" + Fore.RESET)


if __name__ == "__main__":
    start()