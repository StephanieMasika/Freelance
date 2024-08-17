import numpy as np

def iterated_deletion_of_dominated_strategies(payoff_matrix):
    """
    Performs iterated deletion of dominated strategies for a two-player normal form game.

    Parameters:
        payoff_matrix (numpy.ndarray): A 3D array representing the payoff matrix of the game.
                                        Dimensions: (num_rows, 2, num_strategies)
                                        - num_rows: number of possible combinations of actions
                                        - 2: number of players
                                        - num_strategies: number of strategies for each player

    Returns:
        list: A list of surviving strategies after iterated deletion of dominated strategies.
    """
    num_strategies = payoff_matrix.shape[2]
    dominated_strategies = set()  # Keep track of dominated strategies

    # Iterate through each player
    for player in range(2):
        # Iterate through each strategy of the current player
        for strategy in range(num_strategies):
            # Skip if the strategy has already been dominated
            if strategy in dominated_strategies:
                continue

            # List of other strategies for the current player
            other_strategies = [s for s in range(num_strategies) if s != strategy]

            # Check if the current strategy is dominated
            if is_dominated(payoff_matrix, player, strategy, other_strategies):
                dominated_strategies.add(strategy)  # Add the strategy to dominated strategies

    # Return surviving strategies
    return [s for s in range(num_strategies) if s not in dominated_strategies]

def is_dominated(payoff_matrix, player, strategy, other_strategies):
    """
    Checks if a strategy is dominated by other strategies for a player.

    Parameters:
        payoff_matrix (numpy.ndarray): A 3D array representing the payoff matrix of the game.
        player (int): The player for whom dominance is checked.
        strategy (int): The strategy to be checked for dominance.
        other_strategies (list): List of other strategies for the player.

    Returns:
        bool: True if the strategy is dominated, False otherwise.
    """
    # Iterate through other strategies
    for other_strategy in other_strategies:
        # Check if the current strategy dominates the other strategy
        if not dominates(payoff_matrix, player, strategy, other_strategy):
            return False  # If not dominated, return False
    return True  # If dominated by all other strategies, return True

def dominates(payoff_matrix, player, strategy, other_strategy):
    """
    Checks if a strategy dominates another strategy for a player.

    Parameters:
        payoff_matrix (numpy.ndarray): A 3D array representing the payoff matrix of the game.
        player (int): The player for whom dominance is checked.
        strategy (int): The strategy to be checked for dominance.
        other_strategy (int): The other strategy to be checked against.

    Returns:
        bool: True if the strategy dominates the other strategy, False otherwise.
    """
    # Iterate through rows (combinations of actions)
    for row in range(payoff_matrix.shape[0]):
        # Check if the payoff for the current strategy is always greater than the other strategy
        if payoff_matrix[row][player][strategy] < payoff_matrix[row][player][other_strategy]:
            return False  # If not dominates, return False
    return True  # If dominates in all cases, return True

# Example usage:
payoff_matrix = np.array([[[3, 2], [0, 0]], [[2, 3], [0, 0]]])  # Example payoff matrix
surviving_strategies = iterated_deletion_of_dominated_strategies(payoff_matrix)
print("Surviving strategies after iterated deletion of dominated strategies:", surviving_strategies)
