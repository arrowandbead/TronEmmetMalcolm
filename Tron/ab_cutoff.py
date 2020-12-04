def alpha_beta_cutoff(asp, cutoff_ply, eval_func):
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Inputs:
            asp - an AdversarialSearchProblem
            cutoff_ply- an Integer that determines when to cutoff the search
                    and use eval_func.
                    For example, when cutoff_ply = 1, use eval_func to evaluate
                    states that result from your first move. When cutoff_ply = 2, use
                    eval_func to evaluate states that result from your opponent's
                    first move. When cutoff_ply = 3 use eval_func to evaluate the
                    states that result from your second move.
                    You may assume that cutoff_ply > 0.
            eval_func - a function that takes in a GameState and outputs
                    a real number indicating how good that state is for the
                    player who is using alpha_beta_cutoff to choose their action.
                    You do not need to implement this function, as it should be provided by
                    whomever is calling alpha_beta_cutoff, however you are welcome to write
                    evaluation functions to test your implemention. The eval_func we provide
        does not handle terminal states, so evaluate terminal states the
        same way you evaluated them in the previous algorithms.

    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    start = asp.get_start_state()
    player = start.player_to_move()
    #bestMove =  alpha_beta_cutoff_recursive(start, asp, -math.inf, math.inf, player, 0, cutoff_ply, eval_func)
    bestMove = maximize(start, asp, -math.inf, math.inf, player, 0, cutoff_ply, eval_func)

    return bestMove

def maximize(state, asp, alpha, beta, ogPlayer, depth, cutoff, eval_func):
    if(asp.is_terminal_state(state)):
        return(asp.evaluate_state(state)[ogPlayer])
    elif depth == cutoff:
        eVAL = eval_func(state)
        return(eVAL)
    else:
        bestVal = -math.inf
        bestMove = None
        for mv in asp.get_available_actions(state):
            result = minimize(asp.transition(state, mv), asp, alpha, beta, ogPlayer, depth+1, cutoff, eval_func)
            if(result > bestVal):
                bestVal = result
                bestMove = mv
            if(bestVal >= beta):
                if(depth==0):
                    return mv
                return bestVal
            alpha = max(bestVal, alpha)
        if(depth==0):
            return bestMove
        return bestVal

def minimize(state, asp, alpha, beta, ogPlayer, depth, cutoff, eval_func):
    if(asp.is_terminal_state(state)):
        return(asp.evaluate_state(state)[ogPlayer])
    elif depth == cutoff:
        eVAL = eval_func(state)
        return(eVAL)
    else:
        bestVal = math.inf
        for mv in asp.get_available_actions(state):
            result = maximize(asp.transition(state,mv), asp, alpha, beta, ogPlayer, depth + 1, cutoff, eval_func)
            bestVal = min(result, bestVal)
            if(bestVal <= alpha):
                return bestVal
            beta = min(bestVal, beta)
        return bestVal
