# ivan_lee.py
# Chess AI Agent by Ivan Lee
# Enhanced Q-Learning Reinforcement Learning Agent

from agent_interface import Agent
import chess
import numpy as np
import random
import time
from collections import defaultdict

class IvanLeeAgent(Agent):
    def __init__(self, board, color):
        super().__init__(board, color)
        
        # Enhanced learning parameters
        self.learning_rate = 0.3  # Higher learning rate
        self.discount_factor = 0.95
        self.epsilon = 0.4  # Higher exploration initially
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        
        # Q-table for state-action values
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Piece values
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
        self.last_state = None
        self.last_move = None

    def board_to_features(self, board):
        """
        Simplified but more effective state representation
        """
        features = []
        
        # 1. Material balance
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
        
        # 2. Mobility (number of legal moves)
        mobility = len(list(board.legal_moves))
        features.append(mobility)
        
        # 3. King safety (distance from center)
        king_square = board.king(self.color)
        if king_square is not None:
            king_rank = chess.square_rank(king_square)
            king_file = chess.square_file(king_square)
            king_safety = min(king_rank, 7 - king_rank) + min(king_file, 7 - king_file)
            features.append(king_safety)
        else:
            features.append(0)
        
        # 4. Center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        center_control = 0
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == self.color:
                center_control += 1
            elif piece and piece.color != self.color:
                center_control -= 1
        features.append(center_control)
        
        # 5. Check status
        check_bonus = 1 if board.is_check() else 0
        features.append(check_bonus)
        
        return tuple(features)

    def evaluate_move(self, board, move):
        """
        Evaluate a specific move for immediate rewards
        """
        reward = 0
        
        # Capture bonus
        if board.is_capture(move):
            captured_piece = board.piece_type_at(move.to_square)
            if captured_piece:
                reward += self.piece_values.get(captured_piece, 0) * 0.5
        
        # Check bonus
        board.push(move)
        if board.is_check():
            reward += 3
        board.pop()
        
        # Center control bonus
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        if move.to_square in center_squares:
            reward += 1
        
        # King safety
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KING:
            total_pieces = len([p for p in board.piece_map().values()])
            if total_pieces > 20:  # Opening
                reward -= 2
        
        return reward

    def choose_move(self, board, legal_moves):
        """
        Choose move using epsilon-greedy with move evaluation
        """
        state = self.board_to_features(board)
        
        if random.random() < self.epsilon:
            # Explore: choose random move
            return random.choice(legal_moves)
        else:
            # Exploit: choose best move based on Q-values and evaluation
            best_move = None
            best_value = float('-inf')
            
            for move in legal_moves:
                # Get Q-value for this state-action pair
                state_action_key = (state, str(move))
                q_value = self.q_table[state][state_action_key]
                
                # Add immediate move evaluation
                move_eval = self.evaluate_move(board, move)
                total_value = q_value + move_eval
                
                if total_value > best_value:
                    best_value = total_value
                    best_move = move
            
            # If no good move found, choose randomly
            if best_move is None:
                return random.choice(legal_moves)
            
            return best_move

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table with immediate rewards
        """
        state_action_key = (state, str(action))
        
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
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        current_state = self.board_to_features(board)
        
        # Choose move
        chosen_move = self.choose_move(board, legal_moves)
        
        # Learn from immediate reward
        if hasattr(self, 'last_state') and hasattr(self, 'last_move'):
            immediate_reward = self.evaluate_move(board, chosen_move)
            self.update_q_table(self.last_state, self.last_move, immediate_reward, current_state)
        
        # Store for next move
        self.last_state = current_state
        self.last_move = chosen_move
        
        # Time limit check
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            chosen_move = random.choice(legal_moves)
        
        return chosen_move

    def learn_from_game(self, final_board):
        """
        Learn from game outcome
        """
        if not hasattr(self, 'last_state') or not hasattr(self, 'last_move'):
            return
        
        # Calculate final reward
        if final_board.is_checkmate():
            if final_board.turn == self.color:
                reward = -20  # We lost
            else:
                reward = 20   # We won
        elif final_board.is_stalemate():
            reward = 0
        else:
            # Material-based reward
            material_balance = 0
            for square in chess.SQUARES:
                piece = final_board.piece_at(square)
                if piece:
                    value = self.piece_values.get(piece.piece_type, 0)
                    if piece.color == self.color:
                        material_balance += value
                    else:
                        material_balance -= value
            reward = material_balance * 0.1
        
        # Update Q-table
        self.update_q_table(self.last_state, self.last_move, reward, None)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Clear history
        self.last_state = None
        self.last_move = None