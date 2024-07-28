import subprocess
import sys
import time
import csv

# Path to Stockfish executable
# TODO: remove hardcoded path
STOCKFISH_PATH = "Stockfish_bin/stockfish-windows-x86-64/stockfish/stockfish-windows-x86-64.exe"

def get_eval_score(response):
    try:
        eval_score = int(response.split(" ")[9])/100
    except (IndexError, ValueError):
        pass

    return eval_score

def main():
    stockfish = subprocess.Popen(
        STOCKFISH_PATH,
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    
    # Send UCI command to confirm engine identity
    print("send: isready\\n")
    stockfish.stdin.write("isready\n")
    print("send: uci\\n")
    stockfish.stdin.write("uci\n")
    stockfish.stdin.flush()
    
    # Wait for engine to respond with identification
    while True:
        response = stockfish.stdout.readline().strip()
        print(f"recv: {response}")
        if response == "uciok":
            break
    
    field_names = ('FEN', 'ST_1', 'ST_5', 'ST_15')

    with open(f'Stockfish.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        
        # Write header
        writer.writerow(field_names) 
    

        for line in sys.stdin:
            print(f"inp: line->'{line}'")
            fen_position = line.strip()

            if fen_position == "":
                continue

            if fen_position == "end":
                break

            print(f"send: position fen {fen_position}\\n")
            stockfish.stdin.write(f"position fen {fen_position}\n")
            stockfish.stdin.flush()

            print("send: go depth 15\\n")
            stockfish.stdin.write("go depth 15\n")
            stockfish.stdin.flush()

            # evaluation_score  = None
            while True:
                response = stockfish.stdout.readline().strip()
                print(f"recv: {response}")
                if response.startswith("info depth 1 "):
                    evaluation_score1 = get_eval_score(response)
                elif response.startswith("info depth 5 "):
                    evaluation_score5 = get_eval_score(response)
                elif response.startswith("info depth 15 "):
                    evaluation_score15 = get_eval_score(response)
                elif response.startswith("bestmove"):
                    break
            
            
            # writer.writerow((fen_position, evaluation_score1, evaluation_score5, evaluation_score15)) 
            writer.writerow((fen_position, f'"{evaluation_score1}"', f'"{evaluation_score5}"', f'"{evaluation_score15}"')) 
            print(f"XXXX! Evaluation for position '{fen_position}': {evaluation_score15/100}")
        
        
        stockfish.stdin.write("quit\n")
        stockfish.stdin.flush()
        stockfish.terminate()

if __name__ == "__main__":
    main()