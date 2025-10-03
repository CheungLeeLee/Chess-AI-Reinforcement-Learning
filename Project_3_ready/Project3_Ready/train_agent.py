"""
train_agent.py
Simple training script for Ivan Lee's Chess AI Agent
"""

from ivan_lee import IvanLeeAgent
from random_agent import RandomAgent
from greedy_agent import GreedyAgent
from game_driver import run_game
import chess

def train_agent():
    """Train the agent against different opponents"""
    print("=== Ivan Lee Agent Training ===")
    
    # Create agent
    board = chess.Board()
    agent = IvanLeeAgent(board, chess.WHITE)
    
    print(f"Initial Q-table size: {len(agent.q_table)}")
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    
    # Train against Random Agent (easier)
    print("\n=== Training against Random Agent ===")
    wins1, losses1, draws1 = train_against_opponent(agent, RandomAgent, 100)
    
    # Train against Greedy Agent (harder)
    print("\n=== Training against Greedy Agent ===")
    wins2, losses2, draws2 = train_against_opponent(agent, GreedyAgent, 100)
    
    # Test final performance
    print("\n=== Final Performance Test ===")
    test_performance(agent, RandomAgent, "Random Agent", 20)
    test_performance(agent, GreedyAgent, "Greedy Agent", 20)
    
    return agent

def train_against_opponent(agent, opponent_class, num_games):
    """Train agent against a specific opponent"""
    print(f"Training against {opponent_class.__name__} for {num_games} games...")
    
    wins = 0
    losses = 0
    draws = 0
    
    for game_num in range(num_games):
        board = chess.Board()
        
        # Reset agent state
        agent.last_state = None
        agent.last_move = None
        
        # Play game
        move_count = 0
        while not board.is_game_over() and move_count < 200:
            if board.turn == chess.WHITE:
                move = agent.make_move(board, 2.0)
            else:
                # Opponent's move
                opponent = opponent_class(board, chess.BLACK)
                move = opponent.make_move(board, 2.0)
            
            if move:
                board.push(move)
                move_count += 1
            else:
                break
        
        # Learn from game
        agent.learn_from_game(board)
        
        # Count results
        result = board.result()
        if result == "1-0":  # White won
            wins += 1
        elif result == "0-1":  # Black won
            losses += 1
        else:  # Draw
            draws += 1
        
        if (game_num + 1) % 20 == 0:
            print(f"Game {game_num + 1}: Wins={wins}, Losses={losses}, Draws={draws}, Epsilon={agent.epsilon:.3f}")
    
    print(f"Training complete! Wins={wins}, Losses={losses}, Draws={draws}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    
    return wins, losses, draws

def test_performance(agent, opponent_class, opponent_name, num_games):
    """Test agent performance against an opponent"""
    print(f"\n=== Testing vs {opponent_name} ({num_games} games) ===")
    
    wins = 0
    losses = 0
    draws = 0
    
    for i in range(num_games):
        final_board, _ = run_game(agent.__class__, opponent_class, 2.0)
        result = final_board.result()
        
        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1
        
        if (i + 1) % 5 == 0:
            print(f"Game {i+1}: Wins={wins}, Losses={losses}, Draws={draws}")
    
    win_rate = wins / num_games
    print(f"\nResults vs {opponent_name}:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win rate: {win_rate:.1%}")
    
    return win_rate

if __name__ == "__main__":
    agent = train_agent()
    
    print(f"\n=== Training Complete ===")
    print(f"Final Q-table size: {len(agent.q_table)}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print("Your agent is now trained and ready for the tournament!")
