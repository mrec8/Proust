"""
Module to interact with the Jericho environment.
"""
import os
import jericho
import logging
from typing import Dict, List, Tuple, Optional, Any

class JerichoEnvironment:
    """
    Wrapper to interact with interactive fiction games through Jericho directly.
    """
    
    def __init__(self, game_name: str, seed: int = 0):
        """
        Initializes the Jericho environment with the specified game.
        
        Args:
            game_name: Name of the game to load or path to the game file
            seed: Seed for reproducibility (not used in direct Jericho implementation)
        """
        self.logger = logging.getLogger(__name__)
        
        # Try to find the game file
        self.game_path = self._find_game_file(game_name)
        if not self.game_path:
            raise FileNotFoundError(f"Game not found: {game_name}")
        
        # Initialize the Jericho environment directly
        self.env = jericho.FrotzEnv(self.game_path)
        
        # Store information about the current game
        self.game_name = game_name
        
        # Get dictionary details - adapting to the current API
        try:
            # For newer Jericho versions
            self.max_word_length = 20  # Default value if not available
            self.vocab = self.env.get_dictionary()
            if isinstance(self.vocab, list) and self.vocab and hasattr(self.vocab[0], 'decode'):
                # Convert binary strings to text if needed
                self.vocab = [word.decode('cp1252') for word in self.vocab]
        except AttributeError:
            # Fall back to safe defaults if methods don't exist
            self.logger.warning("Could not access dictionary methods, using defaults")
            self.max_word_length = 20
            self.vocab = []
        
        # History and state
        self.steps = 0
        self.max_steps = 100  # Default value, can be updated
        self.score = 0
        self.done = False
        self.history = []
        
        self.logger.info(f"Jericho environment initialized with game: {game_name}")
    
    def _find_game_file(self, game_name: str) -> Optional[str]:
        """
        Tries to find the game file.
        
        Args:
            game_name: Name of the game to find or direct path
            
        Returns:
            Path to the game file or None if not found
        """
        # If game_name is a direct path to a file
        if os.path.isfile(game_name):
            self.logger.info(f"Found direct file: {game_name}")
            return game_name
        
        # Common extensions for interactive fiction games
        extensions = ['.z1', '.z2', '.z3', '.z4', '.z5', '.z6', '.z7', '.z8', '']
        
        # Common locations to look for game files
        search_paths = [
            '.',  # Current directory
            './games/roms',  # Project's game ROMs directory
            os.path.expanduser('~/.jericho/roms'),  # User's Jericho directory
            os.path.join(os.path.dirname(jericho.__file__), 'frotz', 'roms')  # Jericho package directory
        ]
        
        # Try to find the game file with exact name first
        for path in search_paths:
            if os.path.exists(path):
                for ext in extensions:
                    full_path = os.path.join(path, f"{game_name}{ext}")
                    if os.path.exists(full_path):
                        self.logger.info(f"Found game file at: {full_path}")
                        return full_path
        
        # If not found, return None
        self.logger.error(f"Could not find game file for: {game_name}")
        self.logger.info(f"Searched in: {search_paths}")
        return None
    
    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment and returns the initial observation.
        
        Returns:
            Initial state with observation, inventory, score, etc.
        """
        # Reset is handled differently in newer Jericho
        try:
            obs, info = self.env.reset()
        except (TypeError, ValueError):
            # For older Jericho versions
            self.env.reset()
            obs, _, _, info = self.env.step('')  # Empty command to get initial observation
        
        self.steps = 0
        self.history = []
        
        # Extract score from info if available
        if isinstance(info, dict) and 'score' in info:
            self.score = info['score']
        else:
            # Try to extract score from observation
            self.score = 0
        
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
        try:
            # For newer Jericho versions
            observation, reward, done, info = self.env.step(action)
        except ValueError:
            # For older Jericho versions that don't return info
            observation, reward, done = self.env.step(action)
            info = {}
        
        # Increment the step counter
        self.steps += 1
        
        # Update score from info if available
        if isinstance(info, dict) and 'score' in info:
            new_score = info['score']
            reward = new_score - self.score
            self.score = new_score
        else:
            # Use the provided reward
            self.score += reward
            
        self.done = done
        
        # Get inventory
        inventory = self._get_inventory()
        
        # Create the enriched state
        state = {
            'observation': observation,
            'inventory': inventory,
            'score': self.score,
            'moves': self.steps,
            'game_over': done
        }
        
        # Save the action and observation in the history
        self.history.append({
            'action': action,
            'observation': observation,
            'score': self.score
        })
        
        # Log step information with more detail
        self.logger.info(f"Action '{action}' produced response: '{observation}'")
        
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
        # Save the current state before executing inventory command
        try:
            state = self.env.get_state()
            
            # Execute the inventory command
            obs, _, _, _ = self.env.step("inventory")
            
            # Restore the previous state to avoid advancing the game
            self.env.set_state(state)
            
            return obs
        except (AttributeError, TypeError):
            # If get_state/set_state not available, use a basic approach
            self.logger.warning("State saving not available, inventory command may advance the game")
            try:
                # Try to execute inventory command without saving state
                obs, _, _, _ = self.env.step("inventory")
                return obs
            except ValueError:
                # For older Jericho that doesn't return info
                obs, _, _ = self.env.step("inventory")
                return obs
    
    def get_valid_actions(self) -> List[str]:
        """
        Returns a list of valid actions in the current state.
        
        Returns:
            List of valid commands
        """
        try:
            return self.env.get_valid_actions()
        except AttributeError:
            # If method not available, return basic commands
            self.logger.warning("get_valid_actions not available, returning basic commands")
            return ["look", "inventory", "north", "south", "east", "west", "up", "down"]
    
    def get_world_state_description(self) -> Dict[str, Any]:
        """
        Generates a rich description of the current world state.
        
        Returns:
            Dictionary with detailed state information
        """
        try:
            # Save the current state
            state = self.env.get_state()
            
            # Execute several commands to get more information about the state
            look_obs, _, _, _ = self.env.step("look")
            inv_obs = self._get_inventory()
            
            # Restore the previous state
            self.env.set_state(state)
        except (AttributeError, TypeError):
            # If state saving not available, use what we have
            self.logger.warning("State saving not available, using current observation as room description")
            look_obs = self.history[-1]['observation'] if self.history else "No description available"
            inv_obs = self._get_inventory()
        
        return {
            'room_description': look_obs,
            'inventory': inv_obs,
            'score': self.score,
            'moves': self.steps,
            'history': self.history[-5:] if len(self.history) > 5 else self.history  # last 5 actions
        }
    
    def close(self):
        """Closes the environment."""
        if hasattr(self.env, 'close'):
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
            'vocab_size': len(self.vocab) if self.vocab else 0,
            'walkthrough_available': hasattr(self.env, 'get_walkthrough'),
            'steps': self.steps,
            'max_steps': self.max_steps,
            'score': self.score
        }