import numpy as np
import json
import numpy as np
from treys import Card
import pandas as pd
from termcolor import colored

def print_color(my_string , color = 'red'):
    if color == "black" or color == "white":
        if color == "black":
            color = "grey"
        print(colored(my_string, color, attrs=['bold', 'reverse' ]) , end='')
    else:
        print(colored(my_string, color, attrs=['bold', ]) , end='')




def eval_game(game_dict, true_game : list , verbose = True):
    """
    Scores the predicted game labels with the true game labels
    Parameters
    ----------
    game_dict : dict with the cards and chips detected for the game
    true_game : list of the cards and chips in the true game
    verbose : if True print the scores achieved
    ----------
    Returns
    final_score : float with the score achieved in the three task of number, suit and chip identification
    errors : list of bool with the errors in the three task of number, suit and chip identification
    """


    #list to dict, with THOSE headers
    header_dict = ['T1', 'T2', 'T3' , 'T4' ,'T5' , 'P11' , 'P12' , 'P21' , 'P22' , 'P31' , 'P32' , 'P41', 'P42' , 'CR' , 'CG' , 'CB' , 'CK' , 'CW']
    game = [ game_dict[key] for key in header_dict]

    print("Estimated game")
    print(game)
    print("True game")
    print(true_game)
    print("\n")
    
    def parse_game(game : list):
        cards = game[:13]
        chips = game[13:]

        card_numbers = [card[:-1] for card in cards]
        card_suits   = [card[-1:].lower() for card in cards]
        chips        = [int(chip) for chip in chips]


        return np.array(card_numbers), np.array(card_suits), np.array(chips)
    
    
    numbers     ,   suits   , chips      = parse_game(game)
    true_numbers, true_suits, true_chips = parse_game(true_game)
    weight_chip = np.sum(true_chips) + 1
    weight_chip = np.minimum(1/weight_chip , .2)

    errors = list(1 - np.logical_and(numbers== true_numbers , suits == true_suits))
    errors += list(chips - true_chips)
    
    

    score_num   = np.mean(numbers == true_numbers)
    score_suits = np.mean(suits == true_suits) 
    score_chips = np.sum(np.abs(chips - true_chips)*weight_chip)
    score_chips = np.maximum(0 , 1- score_chips)
    final_score = (score_num + score_suits + score_chips)/3
                                
    if verbose:
        print("             \tscores")
        print(f"Card number \t{round(score_num,2)} %" )
        print(f"Card suit   \t{round(score_suits,2)} %" )
        print(f"Chips       \t{round(score_chips,2)} %" )
        print("\n")
        print(f"FINAL score \t{round(final_score,2)} %")
    return final_score ,  errors 

    




def print_game(game_dict , true_game ):
    """
    Pretty prints the predicted game labels with the true game labels
    Parameters
    ----------
    game_dict : dict with the cards and chips detected for the game_id
    true_game : list of the cards and chips in the true game
    ----------
    Returns 
    None
    """
    #list to dict, with THOSE headers
    header_dict = ['T1', 'T2', 'T3' , 'T4' ,'T5' , 'P11' , 'P12' , 'P21' , 'P22' , 'P31' , 'P32' , 'P41', 'P42' , 'CR' , 'CG' , 'CB' , 'CK' , 'CW']
    game = [ game_dict[key] for key in header_dict]
    
    score, errors_game =  eval_game(game_dict, true_game , verbose = False)

    def handle_empty(card):
        if card == '0' or card == 0:
            return None
        card2 = card[:-1] + card[-1:].lower()
        return Card.new(card2)

    def parse_pretty_hand(game):
        board = game[:5]
        all_players = [ game[5+i*2:5+(i+1)*2] for i in range(4)]
        chips = game[13:]
        
        board = [  handle_empty(ci)  for ci in board]
        hands = [ [handle_empty(ci)  for ci in pi ] for pi in all_players]
        chips     = [int(chip) for chip in chips]
        
        return hands , board ,chips

    def print_hand(my_hand):
        #print(" --- " , my_hand)
        if my_hand[0] != None:
            print(Card.print_pretty_cards(my_hand))
        else :
            print(my_hand)
    
    hands_found, board_found ,chips_found = parse_pretty_hand(game)
    hands_true , board_true  ,chips_true  = parse_pretty_hand(true_game)
    

    #### Printing game begins
    
    #printing table cards
    print("__"*20)

    print("Table " ,end = '')
    errors_table = errors_game[0:5]
    if np.sum(errors_table )!=0:

        print_color("mistakes in cards : "  , 'red')
        print(list(np.where(errors_table)[0] ))
        print("True table")
        print(Card.print_pretty_cards(board_true ))
        print("Found table")
        print(Card.print_pretty_cards(board_found ))
    else :
        print_color("found correctly"  , 'green')
        print("\n")
        print(Card.print_pretty_cards(board_found ))
    print('\n')
        
    ##### Printing players cards
    ascii_check = [ "---->" , "xxxxx"]
    #ascii_circle = [🔴,🟢,🔵,⚫,⚪]
    ascii_circle  =[ 0,1,2,3,4]
    errors_hands = errors_game[5:-5]
    for i in range(4):
        sum_error = np.sum(errors_hands[i*2:(i+1)*2])
        right = ascii_check[0] if sum_error == 0 else  ascii_check[1]
        if sum_error  ==0:
            print_color(ascii_check[0] , 'green')
        else:
            print_color(ascii_check[1] , 'red')

        print(f"Player {i+1} " , end= '')
        if sum_error ==0:
            print_color("found correctly" , 'green')
            print_hand(hands_true[i])
        else:
            print_color("error" , 'red')
            print ("\n True hand")
            print_hand(hands_true[i])
            print("Estimated hand")
            print_hand(hands_found[i])                
        print('\n')
    
    #### Printing chips
    chip_names = [ 'Red' , 'Green' , 'Blue' , 'Black' , 'White']
    chip_color = [ 'red' , 'green' , 'blue' , 'black' , 'white']
    for i in range(5):
        diff = chips_found[i] - chips_true[i]
        if diff ==0:
            print_color(ascii_check[0]  , "green")
        else:
            print_color(ascii_check[1]  , "red")
        print_color(f"Chip {chip_names[i]}" , chip_color[i])
        print(f"\tfound: {chips_found[i]}. Count error: ( {diff} )"  , end = '\n')
            
    print("__"*20)
    

    
def eval_listof_games( results_games , game_labels ,game_id = [0,1]):
    """
    Evaluates the results of a list of game_labels
    Parameters
    ----------
    results_games : list of dictionaries with the cards and chips detected for the game_id
    game_labels   : csv file with the true game game_labels
    game_id       : list of the game_id to Evaluates
    ----------
    Returns
    -------
    average_score : float with the final score
    """
    scores = []
    for my_id in game_id:
        print(f"Game {my_id} results")
        game_true = game_labels.iloc[my_id].values[1:]
        game_found = results_games[my_id]
        score , errors = eval_game( game_found ,game_true , verbose = True)
        print("__"*20)
        scores.append(score)
    
    average_score = np.mean(np.array(scores))
    print("Average SCORE = ", average_score)
    return average_score

def debug_listof_games( results_games , game_labels ,game_id = [0,1]):
    """
    Pretty print and detailed evaluation of the results of a dict of game_labels
    Parameters
    ----------
    results_games : list of dictionaries with the cards and chips detected for the game_id
    game_labels   : csv file with the true game game_labels
    game_id       : list of the game_id to Evaluates
    ----------
    Returns
    -------
    """
    for my_id in game_id:
        print(f"Game {my_id} results")
        game_true = game_labels.iloc[my_id].values[1:]
        game_found = results_games[my_id]
        print_game( game_found ,game_true )
        print("__"*20)
    return

def save_results(results= {} , groupid=0):
    """
    Save predctions
    Parameters
    ----------
    results: dict
        Ouput prediction of the process_image function
    group_id: int
        Group id of the students
    """
    file = f'results_group_{str(groupid).zfill(2)}.json'
    with open(file, 'w') as f:
        json.dump(results, f)
    return file


def load_results(file):
    """
    Load predctions
    Parameters
    ----------
    file: str
        File name of the json file with the predictions
    """
    with open(file ,  'r') as f:
        data = json.load(f)
    return data


