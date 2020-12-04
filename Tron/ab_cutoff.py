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
    first = start.player_to_move()
    alpha = float('-inf')
    beta = float('inf')
    moves = 0
    best_action = evaluate_ab_cut(start,first,first,asp,alpha,beta,moves,cutoff_ply,eval_func)
    return best_action[1]

def evaluate_ab_cut(state, player, first, asp, alpha, beta,moves, cutoff_ply,eval_func):
    if asp.is_terminal_state(state):
        return [asp.evaluate_state(state)[first], None]
    if moves == cutoff_ply:
        if player == first:
            return [eval_func(state), None]
        else:
            return [1-eval_func(state), None]
    else:
        #max layer
        if player == first:
            best_action = [float('-inf'),None]
            for action in asp.get_available_actions(state):
                evaluation = evaluate_ab_cut(asp.transition(state,action), get_opp(player), first,asp, alpha,beta, moves+1,cutoff_ply, eval_func)[0]
                if evaluation >= beta:
                    return [evaluation,action]
                alpha = max(evaluation,alpha)
                if evaluation > best_action[0]:
                    best_action = [evaluation, action]

        else:
            #min layer
            best_action = [float('inf'),None]
            for action in asp.get_available_actions(state):
                evaluation = evaluate_ab_cut(asp.transition(state,action), get_opp(player), first,asp, alpha,beta,moves +1, cutoff_ply, eval_func)[0]
                if evaluation <= alpha:
                    return [evaluation,action]
                beta = min(beta,evaluation)
                if evaluation < best_action[0]:
                    best_action = [evaluation, action]

    return best_action
