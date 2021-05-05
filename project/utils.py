import numpy as np


def evaluate_game(pred, cgt, mode_advanced=False):
    """
    Evalutes the accuracy of your predictions. The same function will be used to assess the 
    performance of your model on the final test game.


    Parameters
    ----------
    pred: array of string of shape NxD
        Prediction of the game. N is the number of round (13) and D the number of players (4). Each row 
        is composed of D string. Each string can is composed of 2 charcters [0-9, J, Q, K] + [C, D, H, S].
        If the mode_advanced is False only the rank is evaluated. Otherwise, both rank and colours are 
        evaluated (suits).
    cgt: array of string of shape NxD
        Ground truth of the game. Same format as the prediciton.
    mode_advanced: bool, optional
        Choose the evaluation mode
        
    Returns
    -------
    accuracy: float
        Accuracy of the prediciton wrt the ground truth. Number of correct entries divided by 
        the total number of entries.
    """
    if pred.shape != cgt.shape:
        raise Exception("Prediction and ground truth sould have same shape.")
    
    if mode_advanced:
        # Full performance of the system. Cards ranks and colours.
        return (pred == cgt).mean()
    else:
        # Simple evaluation based on cards ranks only
        cgt_simple = np.array([v[0] for v in cgt.flatten()]).reshape(cgt.shape)
        pred_simple = np.array([v[0] for v in pred.flatten()]).reshape(pred.shape)
        return (pred_simple == cgt_simple).mean()
    
    
def print_results(rank_colour, dealer, pts_standard, pts_advanced):
    """
    Print the results for the final evaluation. You NEED to use this function when presenting the results on the 
    final exam day.
    
    Parameters
    ----------
    rank_colour: array of string of shape NxD
        Prediction of the game. N is the number of round (13) and D the number of players (4). Each row 
        is composed of D string. Each string can is composed of 2 charcters [0-9, J, Q, K] + [C, D, H, S].
    dealer: list of int
        Id ot the players that were selected as dealer ofr each round.
    pts_standard: list of int of length 4
        Number of points won bay each player along the game with standard rules.
    pts_advanced: list of int of length 4
        Number of points won bay each player along the game with advanced rules.
    """
    print('The cards played were:')
    print(pp_2darray(rank_colour))
    print('Players designated as dealer: {}'.format(dealer))
    print('Players points (standard): {}'.format(pts_standard))
    print('Players points (advanced): {}'.format(pts_advanced))
    
    
def pp_2darray(arr):
    """
    Pretty print array
    """
    str_arr = "[\n"
    for row in range(arr.shape[0]):
        str_arr += '[{}], \n'.format(', '.join(["'{}'".format(f) for f in arr[row]]))
    str_arr += "]"
    return str_arr
