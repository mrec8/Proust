"""
Module to interact with the Jericho environment.
"""
import os
import gym
import jericho
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

class JerichoEnvironment:
    """
    Wrapper to interact with interactive fiction games through Jericho.
    """
    
    def __init__(self, game_name: str, seed: int = 0):
        """
        Initializes the Jericho environment with the specified game.
        
        Args:
            game_name: Name of the game to load
            seed: Seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        
        # Build the path to the game file
        self.game_path = os.path.join(jericho.DATA_PATH, f"{game_name}.z5")
        if not os.path.exists(self.game_path):
            self.game_path = os.path.join(jericho.DATA_PATH, f"{game_name}.z6")
            if not os.path.exists(self.game_path):
                self.game_path = os.path.join(jericho.DATA_PATH, f"{game_name}.z8")
                if not os.path.exists(self.game_path):
                    raise FileNotFoundError(f"Game not found: {game_name}")
        
        # Initialize the Jericho environment
        self.env = gym.make(f"jericho/{game_name}-v0")
        self.env.seed(seed)
        
        # Store information about the current game
        self.game_name = game_name
        self.max_word_length = self.env.get_dictionary_max_length()
        self.vocab = self.env.get_dictionary()
        
        # History and state
        self.steps = 0
        self.max_steps = 100  # Default value, can be updated
        self.score = 0
        self.done = False
        self.history = []
        
        self.logger.info(f"Jericho environment initialized with game: {game_name}")
    
    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment and returns the initial observation.
        
        Returns:
            Initial state with observation, inventory, score, etc.
        """
        obs, info = self.env.reset()
        self.steps = 0
        self.history = []
        self.score = info['score']
        self.done = False
        
        # Get inventory
        inventory = self._get_inventory()
        
        # Create an enriched state
        state = {
            'observation': obs,
            'inventory': inventory,
            'score': self.score,
            'moves': self.steps,
            'game_over': self.done
        }
        
        self.history.append({
            'action': 'RESET',
            'observation': obs,
            'score': self.score
        })
        
        self.logger.info("Environment reset")
        return state
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes an action in the environment and returns the new state.
        
        Args:
            action: Text command to execute
            
        Returns:
            Tuple (state, reward, done, info) with the new state, reward, 
            whether it has finished and additional information
        """
        # Execute the action
        observation, score, done, info = self.env.step(action)
        
        # Increment the step counter
        self.steps += 1
        
        # Calculate the reward (score difference)
        reward = score - self.score
        self.score = score
        self.done = done
        
        # Get inventory
        inventory = self._get_inventory()
        
        # Create the enriched state
        state = {
            'observation': observation,
            'inventory': inventory,
            'score': score,
            'moves': self.steps,
            'game_over': done
        }
        
        # Save the action and observation in the history
        self.history.append({
            'action': action,
            'observation': observation,
            'score': score
        })
        
        # Log step information
        self.logger.debug(f"Step {self.steps}: Action='{action}', Score={score}, Reward={reward}, Done={done}")
        
        # If we reach the maximum number of steps, we finish
        if self.steps >= self.max_steps:
            done = True
            info['reason'] = 'max_steps'
            self.logger.info(f"Reached maximum steps ({self.max_steps}). Ending episode.")
        
        return state, reward, done, info
    
    def _get_inventory(self) -> str:
        """
        Gets the player's current inventory.
        
        Returns:
            Text describing the inventory
        """
        # Execute the inventory command
        obs, _, _, _ = self.env.step("inventory")
        
        return obs
    
    def get_valid_actions(self) -> List[str]:
        """
        Returns a list of valid actions in the current state.
        
        Returns:
            List of valid commands
        """
        return self.env.get_valid_actions()
    
    def get_world_state_description(self) -> Dict[str, Any]:
        """
        Generates a rich description of the current world state.
        
        Returns:
            Dictionary with detailed state information
        """
        # Execute several commands to get more information about the state
        look_obs, _, _, _ = self.env.step("look")
        inv_obs, _, _, _ = self.env.step("inventory")
        
        return {
            'room_description': look_obs,
            'inventory': inv_obs,
            'score': self.score,
            'moves': self.steps,
            'history': self.history[-5:] if len(self.history) > 5 else self.history  # last 5 actions
        }
    
    def close(self):
        """Closes the environment."""
        self.env.close()
        self.logger.info("Environment closed")
    
    def get_walkthrough(self) -> Optional[List[str]]:
        """
        Returns the walkthrough for the game if available.
        
        Returns:
            List of walkthrough actions or None if not available
        """
        if hasattr(self.env, 'get_walkthrough'):
            return self.env.get_walkthrough()
        return None
    
    def get_game_info(self) -> Dict[str, Any]:
        """
        Returns information about the game.
        
        Returns:
            Dictionary with game information
        """
        return {
            'game_name': self.game_name,
            'max_word_length': self.max_word_length,
            'vocab_size': len(self.vocab),
            'walkthrough_available': hasattr(self.env, 'get_walkthrough'),
            'steps': self.steps,
            'max_steps': self.max_steps,
            'score': self.score
        }