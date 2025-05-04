"""
Critical Agent for self-verification of task success.
"""
import os
import yaml
import json
import re
from typing import Dict, List, Tuple, Any, Optional

from utils.llm_interface import LLMInterface
from environment.observation_parser import ObservationParser

class CriticAgent:
    """
    Agent that evaluates task success and provides feedback.
    """
    
    def __init__(self, config_path: str, game_config_path: str, llm: LLMInterface):
        """
        Initializes the critic agent.
        
        Args:
            config_path: Path to the configuration file
            game_config_path: Path to the game configuration file
            llm: Interface with the language model
        """
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(game_config_path, 'r') as f:
            self.game_config = yaml.safe_load(f)
        
        self.llm = llm
        
        # Specific configurations
        self.game_name = self.config['environment']['game']
        self.game_specific_config = self.game_config.get(self.game_name, {})
        
        # Critic agent parameters
        self.strictness = self.config['agents']['critic_agent'].get('strictness', 0.8)
        
        # Initialize the observation parser
        self.obs_parser = ObservationParser()
        
        # Evaluation history
        self.evaluation_history = []
    
    def check_task_success(self, task: str, agent_state: Dict[str, Any], 
                          action_history: Optional[List[Tuple[str, str]]] = None) -> Tuple[bool, str]:
        """
        Verifies if a task has been successfully completed.
        
        Args:
            task: Task to evaluate
            agent_state: Current state of the agent
            action_history: Action history (task, action)
            
        Returns:
            Tuple (success, critique)
        """
        # Process observation
        parsed_observation = self.obs_parser.parse_observation(agent_state.get('observation', ''))
        
        # Process inventory
        inventory = agent_state.get('inventory', '')
        parsed_inventory = self.obs_parser.parse_inventory(inventory)
        
        # Extract additional information
        
        # Build verification prompt
        prompt = self._build_verification_prompt(task, parsed_observation, 
                                               parsed_inventory, agent_state, 
                                               action_history)
        
        # Generate evaluation
        response = self.llm.generate(prompt, temperature=0.4)
        
        # Extract evaluation result
        success, critique = self._extract_evaluation_result(response)
        
        # Log evaluation
        self._add_to_history(task, success, critique)
        
        return success, critique
    
    def _build_verification_prompt(self, task: str, parsed_observation: Dict[str, Any], 
                                  parsed_inventory: List[str], agent_state: Dict[str, Any], 
                                  action_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Builds the prompt to verify the success of a task.
        
        Args:
            task: Task to evaluate
            parsed_observation: Processed observation
            parsed_inventory: Processed inventory
            agent_state: Current state of the agent
            action_history: Action history
            
        Returns:
            Prompt for the LLM
        """
        # Extract relevant information
        location = parsed_observation.get('location', '')
        objects = parsed_observation.get('objects', [])
        exits = parsed_observation.get('exits', [])
        entities = parsed_observation.get('entities', [])
        messages = parsed_observation.get('messages', [])
        
        score = agent_state.get('score', 0)
        moves = agent_state.get('moves', 0)
        
        # Build recent history
        recent_history = ""
        if action_history:
            recent_history = "RECENT ACTIONS:\n"
            for i, (t, a) in enumerate(action_history[-5:]):  # Last 5 actions
                recent_history += f"{i+1}. Task: {t} -> Action: {a}\n"
        
        # Build the prompt
        prompt = f"""
        You are an expert critic in interactive fiction games like {self.game_name}.
        Your task is to evaluate whether an agent has successfully completed a specific task.
        You must be precise and strict in your evaluation.
        
        TASK TO EVALUATE:
        {task}
        
        CURRENT STATE OF THE AGENT:
        Location: {location}
        Visible objects: {', '.join(objects) if objects else 'None'}
        Exits: {', '.join(exits) if exits else 'None visible'}
        Entities: {', '.join(entities) if entities else 'None'}
        Messages: {', '.join(messages) if messages else 'None'}
        
        INVENTORY:
        {', '.join(parsed_inventory) if parsed_inventory else 'Empty'}
        
        Score: {score}
        Moves: {moves}
        
        {recent_history}
        
        INSTRUCTIONS:
        1. Evaluate if the task has been successfully completed based on the current state.
        2. Consider changes in inventory, location, visible objects, and messages.
        3. If the task has not been completed, provide constructive critique.
        4. Be strict but fair in your evaluation.
        
        Respond in the following JSON format:
        {{
          "reasoning": "your detailed analysis",
          "success": true/false,
          "critique": "constructive critique if unsuccessful, or empty if successful"
        }}
        """
        
        return prompt
    
    def _extract_evaluation_result(self, response: str) -> Tuple[bool, str]:
        """
        Extracts the evaluation result from the text generated by the LLM.
        
        Args:
            response: Full text generated by the LLM
            
        Returns:
            Tuple (success, critique)
        """
        try:
            # Attempt to extract JSON
            # Find the first '{' and the last '}'
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                result = json.loads(json_str)
                
                success = result.get('success', False)
                critique = result.get('critique', '')
                
                return success, critique
            
        except json.JSONDecodeError:
            # If JSON extraction fails, attempt a pattern-based approach
            pass
        
        # Backup pattern-based approach
        if "success: true" in response.lower():
            return True, ""
        
        # Search for a critique
        critique = ""
        critique_patterns = [
            r"critique: \"(.*?)\"",
            r"critique:(.*?)(?:\"|$)"
        ]
        
        for pattern in critique_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                critique = match.group(1).strip()
                break
        
        return False, critique
    
    def _add_to_history(self, task: str, success: bool, critique: str) -> None:
        """
        Adds an evaluation to the history.
        
        Args:
            task: Evaluated task
            success: Evaluation result
            critique: Provided critique
        """
        self.evaluation_history.append({
            'task': task,
            'success': success,
            'critique': critique
        })
        
        # Limit the size of the history (keep the last 50 evaluations)
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]