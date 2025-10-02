# ivan_lee.py
# Chess AI Agent by Ivan Lee
# Q-Learning Reinforcement Learning Agent
from agent_interface import Agent
import chess
import numpy as np
import random
import time
from collections import defaultdict

class IvanLeeAgent(Agent):
    def __init__(self, board, color):
        super().__init__(board, color)
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Piece values for material evaluation
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Game history for learning
        self.game_history = []
        self.current_state = None

    def board_to_features(self, board):
        """
        Convert chess board to a simple feature vector.
        This is a basic representation - you can improve this!
        """
        features = []
        
        # Material count (simplified)
        material_balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == self.color:
                    material_balance += value
                else:
                    material_balance -= value
        
        features.append(material_balance)
        
        # Center control (simplified)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        center_control = 0
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == self.color:
                center_control += 1
            elif piece and piece.color != self.color:
                center_control -= 1
        
        features.append(center_control)
        
        # King safety (simplified)
        king_square = board.king(self.color)
        if king_square is not None:
            king_rank = chess.square_rank(king_square)
            king_file = chess.square_file(king_square)
            # Prefer king in corners/edges for safety
            king_safety = min(king_rank, 7 - king_rank) + min(king_file, 7 - king_file)
            features.append(king_safety)
        else:
            features.append(0)
        
        # Convert to tuple for hashing
        return tuple(features)

    def evaluate_position(self, board):
        """
        Simple position evaluation function.
        Returns a score from the perspective of self.color.
        """
        score = 0
        
        # Material evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == self.color:
                    score += value
                else:
                    score -= value
        
        # Simple positional bonuses
        if board.is_checkmate():
            if board.turn == self.color:
                score -= 1000  # We're checkmated
            else:
                score += 1000  # We checkmated opponent
        
        if board.is_stalemate():
            score = 0  # Draw
        
        return score

    def get_state_action_key(self, state, move):
        """Create a key for state-action pair"""
        return (state, str(move))

    def choose_move(self, board, legal_moves):
        """
        Choose move using epsilon-greedy policy
        """
        state = self.board_to_features(board)
        
        if random.random() < self.epsilon:
            # Explore: choose random move
            return random.choice(legal_moves)
        else:
            # Exploit: choose best known move
            best_move = None
            best_value = float('-inf')
            
            for move in legal_moves:
                state_action_key = self.get_state_action_key(state, move)
                q_value = self.q_table[state][state_action_key]
                
                if q_value > best_value:
                    best_value = q_value
                    best_move = move
            
            # If no move has been seen before, choose randomly
            if best_move is None:
                return random.choice(legal_moves)
            
            return best_move

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning formula
        """
        state_action_key = self.get_state_action_key(state, action)
        
        # Find max Q-value for next state
        max_next_q = 0
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Q-learning update
        current_q = self.q_table[state][state_action_key]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][state_action_key] = new_q

    def make_move(self, board, time_limit):
        """
        Main method called by the tournament driver
        """
        start_time = time.time()
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Choose move
        chosen_move = self.choose_move(board, legal_moves)
        
        # Store current state for learning
        self.current_state = self.board_to_features(board)
        
        # Ensure we don't exceed time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            # Fallback to random move if we're running out of time
            chosen_move = random.choice(legal_moves)
        
        return chosen_move

    def learn_from_game(self, final_board):
        """
        Learn from the completed game
        This would be called after each game in training
        """
        if not self.game_history:
            return
        
        # Get final reward
        final_score = self.evaluate_position(final_board)
        
        # Simple reward: +1 for win, -1 for loss, 0 for draw
        if final_score > 0:
            reward = 1
        elif final_score < 0:
            reward = -1
        else:
            reward = 0
        
        # Update Q-values for the last move
        if len(self.game_history) > 0:
            last_state, last_action = self.game_history[-1]
            self.update_q_table(last_state, last_action, reward, None)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Clear history for next game
        self.game_history = []