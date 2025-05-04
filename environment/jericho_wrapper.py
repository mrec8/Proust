"""
Module to interact with the Jericho environment.
"""
import os
import gym
import numpy as np
import jericho
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
        # Build the path to the game file
        self.game_path = os.path.join(jericho.DATA_PATH, f"{game_name}.z5")
        if not os.path.exists(self.game_path):
            self.game_path = os.path.join(jericho.DATA_PATH, f"{game_name}.z6")
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
        self.history = []
        self.initial_observation = None
        self.current_observation = None
        self.current_score = 0
        self.done = False
        
    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment and returns the initial observation.
        
        Returns:
            Initial state with observation, inventory, score, etc.
        """
        obs, info = self.env.reset()
        self.steps = 0
        self.history = []
        self.current_score = info['score']
        self.done = False
        
        # Get more complete information
        valid_actions = self.env.get_valid_actions()
        inventory = self._get_inventory()
        
        # Save the initial observation
        self.initial_observation = obs
        self.current_observation = obs
        
        # Create an enriched state
        state = {
            'observation': obs,
            'inventory': inventory,
            'score': self.current_score,
            'moves': self.steps,
            'valid_actions': valid_actions,
            'game_over': self.done
        }
        
        self.history.append({
            'action': 'RESET',
            'observation': obs,
            'score': self.current_score
        })
        
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
        reward = score - self.current_score
        self.current_score = score
        self.done = done
        
        # Update the current observation
        self.current_observation = observation
        
        # Get more information
        valid_actions = self.env.get_valid_actions()
        inventory = self._get_inventory()
        
        # Create the enriched state
        state = {
            'observation': observation,
            'inventory': inventory,
            'score': score,
            'moves': self.steps,
            'valid_actions': valid_actions,
            'game_over': done
        }
        
        # Save the action and observation in the history
        self.history.append({
            'action': action,
            'observation': observation,
            'score': score
        })
        
        # If we reach the maximum number of steps, we finish
        if self.steps >= self.max_steps:
            done = True
            info['reason'] = 'max_steps'
        
        return state, reward, done, info
    
    def _get_inventory(self) -> str:
        """
        Gets the player's current inventory.
        
        Returns:
            Text describing the inventory
        """
        # Save current observation
        current_obs = self.current_observation
        
        # Execute the inventory command
        obs, _, _, _ = self.env.step("inventory")
        
        # Restore the state (Jericho does not affect the state with 'inventory')
        self.current_observation = current_obs
        
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
        current_obs = self.current_observation
        
        # "look" command to see around
        look_obs, _, _, _ = self.env.step("look")
        
        # "inventory" command to see inventory
        inv_obs, _, _, _ = self.env.step("inventory")
        
        # Restore the state
        self.current_observation = current_obs
        
        return {
            'room_description': look_obs,
            'inventory': inv_obs,
            'score': self.current_score,
            'moves': self.steps,
            'history': self.history[-5:] if len(self.history) > 5 else self.history  # last 5 actions
        }
    
    def close(self):
        """Closes the environment."""
        self.env.close()