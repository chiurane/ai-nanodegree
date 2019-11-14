"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

infinity = float('inf')
argmax = max
argmin = min


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def utility(game, player):
    """
    Compute the sum of the distances of legal moves to the center.
    :param player: player
    :param game: game
    :return: the sum of the distances of all legal moves to the center
    """
    # from scipy.spatial import distance
    moves = game.get_legal_moves(player)
    (w, h) = round(game.width / 2.0), round(game.height / 2.0)  # game width and height
    # u = [distance.cityblock((w, h), move) for move in moves]
    u = [float((w - move[0]) ** 2 + (h - move[1]) ** 2) for move in moves]
    return sum(u)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # raise NotImplementedError
    # Lets first check if someone has won or lost
    import math
    if game.is_winner(player):
        return infinity
    elif game.is_loser(player):
        return -infinity
    else:
        # Let g_n be the function represented by a^2 - k*b + c
        # Where:
        #  a is the sum of distances of legal moves from the center
        #  k is some lambda penalty constant
        #  b is the sum of distances of legal moves of opponent from the center
        # c is the distance from centre as given by e^(distance)
        k = 3.0  # some lambda penalty
        # a = utility(game, player)
        # b = utility(game, game.get_opponent(player))
        a = len(game.get_legal_moves(player))
        b = len(game.get_legal_moves(game.get_opponent(player)))
        (w, h) = round(game.width / 2.0), round(game.height / 2.0)
        c = float((w - game.get_player_location(player)[0]) ** 2 + (h - game.get_player_location(player)[1]) ** 2)
        g_n = a ** 2 - k * b + math.exp(c)
        return g_n


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    import math
    # raise NotImplementedError
    # Lets first check if someone has won or lost
    if game.is_winner(player):
        return infinity
    elif game.is_loser(player):
        return -infinity
    else:
        # Let g_n be the function given by (a-b)*((a-b)/(a+b))
        # Where:
        #   a is the sum of the distances of player's legal moves from the center
        #   b is the sum of the distances of opponent's legal moves from the centre
        k = 3.0  # some lambda penalty
        a = utility(game, player)
        b = utility(game, game.get_opponent(player))
        g_n = a ** 2 - k * b + math.exp(
            (len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player)))))
        return g_n


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # raise NotImplementedError
    # raise NotImplementedError
    # Lets first check if someone has won or lost
    if game.is_winner(player):
        return infinity
    elif game.is_loser(player):
        return -infinity
    else:
        # Let g_n be a function given by a(c-d*b) and h_n be given by b*(d-c*a)
        # Where:
        #   a is the sum of the distances of all legal moves to the center for the player
        #   b is the sum of the distances of all legal moves to the centre for the opponent
        #   c is the distance of the current player position to the centre
        #   d is the distance of the opponent position to the centre
        # return g_n - k*h_n
        import math
        k = 3.0
        a = utility(game, player)
        b = utility(game, game.get_opponent(player))
        (w, h) = round(game.width / 2.0), round(game.height / 2.0)  # game width and height
        (x, y) = game.get_player_location(player)  # my location
        (u, v) = game.get_player_location(game.get_opponent(player))  # Opponent location
        c = float((w - x) ** 2 + (h - y) ** 2)
        d = float((w - u) ** 2 + (h - v) ** 2)
        g_n = a * (c - d * b)
        h_n = b * (d - c * a)
        return g_n - k * h_n + math.exp(
            len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        # if self.time_left() < self.TIMER_THRESHOLD: move to min and max
        #    raise SearchTimeout()
        # TODO: finish this function!
        # Based AIMA Textbook example
        def max_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth <= 1:
                return self.score(game, self)

            # Set to a minimum initially
            v = -infinity
            for legal_move in game.get_legal_moves():  # Get all the available legal moves
                v = max(v, min_value(game.forecast_move(legal_move), depth - 1))
            return v

        def min_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth <= 1:
                return self.score(game, self)

            # Set v to a maximum initially
            v = infinity
            for legal_move in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(legal_move), depth - 1))
            return v

        # raise NotImplementedError
        # Do we still have time?
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = -infinity
        best_move = (-1, -1)  # In case we dont get a move, plausible but unlikely

        for legal_move in game.get_legal_moves():
            move_score = min_value(game.forecast_move(legal_move), depth)
            if move_score > best_score:
                best_score = move_score
                best_move = legal_move
        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
        search with alpha-beta pruning. You must finish and test this player to
        make sure it returns a good move before the search time limit expires.
        """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
                result before the time limit expires.

                Modify the get_move() method from the MinimaxPlayer class to implement
                iterative deepening search instead of fixed-depth search.

                **********************************************************************
                NOTE: If time_left() < 0 when this function returns, the agent will
                      forfeit the game due to timeout. You must return _before_ the
                      timer reaches 0.
                **********************************************************************

                Parameters
                ----------
                game : `isolation.Board`
                    An instance of `isolation.Board` encoding the current state of the
                    game (e.g., player locations and blocked cells).

                time_left : callable
                    A function that returns the number of milliseconds left in the
                    current turn. Returning with any less than 0 ms remaining forfeits
                    the game.

                Returns
                -------
                (int, int)
                    Board coordinates corresponding to a legal move; may return
                    (-1, -1) if there are no available legal moves.
                """
        self.time_left = time_left

        # Check if we have time left,  abort if not, otherwise continue
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Initialize best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        # Get all the legal moves here, this is the first attempt to initialise best move if we have any moves.
        moves = game.get_legal_moves()
        if len(moves) == 0:
            return best_move
        else:
            # Even in Alphabeta, a policy of selecting the center is always a good start - good old human wisdom.
            # (w,h) = round(game.width/2.0), round(game.height/2.0)
            # if game.move_is_legal((w,h)):
            #    best_move = (w,h)
            #    return best_move
            best_move = moves[0]

        depth = 0

        # We use a try block to look out for potential timeouts
        try:
            while self.time_left() > self.TIMER_THRESHOLD:
                depth += 1
                # Call Alphabeta as long as we have time left
                best_move = self.alphabeta(game, depth)
        except SearchTimeout:
            pass

        # Return best move from last completed iteration here ...
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
                described in the lectures.

                This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
                https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

                **********************************************************************
                    You MAY add additional methods to this class, or define helper
                         functions to implement the required functionality.
                **********************************************************************

                Parameters
                ----------
                game : isolation.Board
                    An instance of the Isolation game `Board` class representing the
                    current game state

                depth : int
                    Depth is an integer representing the maximum number of plies to
                    search in the game tree before aborting

                alpha : float
                    Alpha limits the lower bound of search on minimizing layers

                beta : float
                    Beta limits the upper bound of search on maximizing layers

                Returns
                -------
                (int, int)
                    The board coordinates of the best move found in the current search;
                    (-1, -1) if there are no legal moves

                Notes
                -----
                    (1) You MUST use the `self.score()` method for board evaluation
                        to pass the project tests; you cannot call any other evaluation
                        function directly.

                    (2) If you use any helper functions (e.g., as shown in the AIMA
                        pseudocode) then you must copy the timer check into the top of
                        each helper function or else your agent will timeout during
                        testing.
                """
        # We check that we have time first ...
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        import operator # For Alphabeta ordering ...

        def max_value(game, depth, alpha, beta):
            """
            Compute max value
            :param game: Isolation.Board
            :param depth: Search deapth
            :param alpha: alpha parameter
            :param beta: beta parameter
            :return: maximum
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth <= 1:
                return self.score(game, self)
            # Set v to a minimum (-infinity)
            v = -infinity  # Set v to a minimum
            # For Alphabeta ordering
            game.get_legal_moves().sort(key=operator.itemgetter(0))
            for legal_move in game.get_legal_moves():
                v = argmax(v, min_value(game.forecast_move(legal_move), depth - 1, alpha, beta))
                alpha = argmax(alpha, v)
                if v >= beta:
                    return v
            return v

        def min_value(game, depth, alpha, beta):
            """
            COmpute the minimum in Alphabeta search
            :param game: Isolation.board
            :param depth: Search depth
            :param alpha: the alpha parameter
            :param beta: the beta parameter
            :return: minimum
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth <= 1:
                return self.score(game, self)
            # Set v to a maximum (+infinity)
            v = infinity
            # For Alphabeta ordering
            game.get_legal_moves().sort(key=operator.itemgetter(0), reverse=True)
            for legal_move in game.get_legal_moves():
                v = argmin(v, max_value(game.forecast_move(legal_move), depth - 1, alpha, beta))
                beta = argmin(beta, v)
                if v <= alpha:
                    return v
            return v

        # Check if we still time left ...
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # We set our best score to -infinity (a minimum)
        best_score = -infinity

        # Beta is set to +infinity
        beta = infinity

        # We want to compute the best course of action
        best_action = None

        # Go through all legal moves, to find the best move using min_value and max_value functions
        for legal_move in game.get_legal_moves():
            v = min_value(game.forecast_move(legal_move), depth, best_score, beta)
            if v > best_score:
                best_score = v
                best_action = legal_move

        # Return the computed best action
        return best_action

