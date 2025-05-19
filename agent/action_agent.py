"""
Action Agent to generate executable commands in the Jericho environment.
"""
import os
import re
import yaml
from typing import Dict, List, Tuple, Set, Any, Optional

from utils.llm_interface import LLMInterface
from environment.observation_parser import ObservationParser
from agent.skill_manager import Skill

class ActionAgent:
    """
    Agent that generates executable commands/actions in the Jericho environment.
    """
    
    def __init__(self, config: Dict[str, Any], game_config: Dict[str, Any], llm: LLMInterface):
        """
        Initializes the action agent.
        
        Args:
            config_path: Path to the configuration file
            game_config_path: Path to the game configuration file
            llm: Interface with the language model
        """
        # Load configurations
        
        self.config = config
        
        self.game_config = game_config
        
        self.llm = llm
        
        # Specific configurations
        self.game_name = self.config['environment']['game']
        self.game_specific_config = self.game_config.get(self.game_name, {})
        
        # Action agent parameters
        self.max_refinement_iterations = self.config['agents']['action_agent'].get('max_refinement_iterations', 3)
        
        # Initialize the observation parser
        self.obs_parser = ObservationParser()
        
        # Store special commands for the game
        self.special_commands = self.game_specific_config.get('special_commands', [])
        
        # Action history
        self.action_history = []
    
    def generate_action(self, task: str, agent_state: Dict[str, Any], critique: str, 
                       skills: Optional[List[Dict[str, Any]]] = None ) -> str:
        """
        Generates an executable command/action for the given task.
        
        Args:
            task: Current task
            agent_state: Current state of the agent
            skills: List of relevant skills from the library
            critique: Critique of the previous action regarding the task
        Returns:
            Executable command/action
        """
        # Process the observation
        parsed_observation = self.obs_parser.parse_observation(agent_state.get('observation', ''))
        
        # Process the inventory
        inventory = agent_state.get('inventory', '')
        parsed_inventory = self.obs_parser.parse_inventory(inventory)
        
        # Build prompt for the LLM
        prompt = self._build_action_generation_prompt(task, parsed_observation, 
                                                     parsed_inventory, agent_state, critique, skills)
        
        # Generate action
        response = self.llm.generate(prompt, temperature=0.7)
        
        # Extract action from the generated text
        action = self._extract_action_from_response(response)
        
        # Log the action in the history
        self._add_to_history(task, action)
        
        return action
    
    def refine_action(self, previous_action: str, task: str, agent_state: Dict[str, Any], 
                     error_message: Optional[str] = None) -> str:
        """
        Refines a previous action that was not successful.
        
        Args:
            previous_action: Previous action to be refined
            task: Current task
            agent_state: Current state of the agent
            error_message: Error message or result of the previous action
            
        Returns:
            Refined command/action
        """
        # Process the observation
        parsed_observation = self.obs_parser.parse_observation(agent_state.get('observation', ''))
        
        # Process the inventory
        inventory = agent_state.get('inventory', '')
        parsed_inventory = self.obs_parser.parse_inventory(inventory)
        
        # Build prompt for refinement
        prompt = self._build_action_refinement_prompt(previous_action, task, 
                                                     parsed_observation, parsed_inventory, 
                                                     agent_state, error_message)
        
        # Generate refined action
        response = self.llm.generate(prompt, temperature=0.7)
        
        # Extract action from the generated text
        action = self._extract_action_from_response(response)
        
        # Log the refined action in the history
        self._add_to_history(task, action, is_refinement=True)
        
        return action
    
    def _build_action_generation_prompt(self, task: str, parsed_observation: Dict[str, Any], 
                                        parsed_inventory: List[str], agent_state: Dict[str, Any], 
                                        critique: str,
                                        skills: Optional[List[Skill]] = None) -> str:
        """
        Builds the prompt to generate an action.
        
        Args:
            task: Current task
            parsed_observation: Processed observation
            parsed_inventory: Processed inventory
            agent_state: Current state of the agent
            critique: Critique of the previous action
            skills: List of relevant skills
            
        Returns:
            Prompt for the LLM
        """
        
        
        # Build relevant skills context
        skills_context = ""
        if skills:
            skills_context = "RELEVANT SKILLS:\n"
            for skill in skills:
                # Access skill attributes using dot notation, not dictionary subscription
                skills_context += f"- {skill.description}\n"
        
        
        
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')

        # Build the prompt
        prompt = f"""You are a top-tier expert in text adventure games. Your task is to generate ONE precise command to achieve the following objective in the game '{self.game_name}'.

        TASK:
        {task}

        CURRENT GAME STATE:
        {observation}

        INVENTORY:
        {inventory}

        CRITIC FEEDBACK (IMPORTANT CONTEXT):
        {critique}

        RELEVANT SKILLS:
        {skills_context if skills else "No relevant skills available."}

        --- GUIDELINES FOR COMMAND GENERATION ---

        1. USEFUL COMMAND PATTERNS:
        - [verb] → look, inventory, wait
        - [verb] [noun] → take leaflet, open mailbox
        - [verb] [noun] [prep] [noun] → put coin in slot
        - [direction] → north, south, east, west, up, down

        2. PROHIBITED PHRASES OR STRUCTURES:
        - NO full sentences or explanations
        - NO vague commands like "explore", "use object", "go to location"
        - NO commands longer than 4 words
        - NO quotes, articles, or redundant modifiers ("take the leaflet" → ❌)

        3. SPECIAL GAME COMMANDS:
        {', '.join(self.special_commands)}

        4. CRITIQUE-DRIVEN ADAPTATION:
        - Do NOT repeat failed actions
        - Use *exactly* the improved verbs or objects recommended
        - Do NOT interact with objects the game can't see
        - Only act if the game's state suggests the object is present and relevant

        --- OUTPUT FORMAT ---

        Your response must:
        - Be a **single command** (1-2 words max)
        - Follow text adventure syntax strictly
        - Match the current task context
        - Contain **no explanations or extra text**

        YOUR COMMAND:
        """
        return prompt
    
    
    def _build_action_refinement_prompt(self, previous_action: str, task: str, 
                                       parsed_observation: Dict[str, Any], 
                                       parsed_inventory: List[str], 
                                       agent_state: Dict[str, Any], 
                                       error_message: Optional[str] = None) -> str:
        """
        Builds the prompt to refine a previous action.
        
        Args:
            previous_action: Previous action to be refined
            task: Current task
            parsed_observation: Processed observation
            parsed_inventory: Processed inventory
            agent_state: Current state of the agent
            error_message: Error message or result of the previous action
            
        Returns:
            Prompt for the LLM
        """
        # Extract relevant information
        location = parsed_observation.get('location', '')
        objects = parsed_observation.get('objects', [])
        exits = parsed_observation.get('exits', [])
        entities = parsed_observation.get('entities', [])
        messages = parsed_observation.get('messages', [])
        
        # Build error context
        error_context = ""
        if error_message:
            error_context = f"""
            RESULT OF THE PREVIOUS ACTION:
            Command: {previous_action}
            Result: {error_message}
            """
        
        # Build the prompt
        prompt = f"""
        You are an expert agent in interactive fiction games like {self.game_name}.
        Your task is to REFINE a previous command that was not successful.
        
        CURRENT TASK:
        {task}
        
        PREVIOUS COMMAND:
        {previous_action}
        
        {error_context}
        
        CURRENT STATE:
        Location: {location}
        Visible objects: {', '.join(objects) if objects else 'None'}
        Exits: {', '.join(exits) if exits else 'None visible'}
        Entities: {', '.join(entities) if entities else 'None'}
        Messages: {', '.join(messages) if messages else 'None'}
        
        INVENTORY:
        {', '.join(parsed_inventory) if parsed_inventory else 'Empty'}
        
        INSTRUCTIONS FOR REFINEMENT:
        1. Analyze why the previous command did not work.
        2. Generate ONE SINGLE alternative command that might work better.
        3. Commands must be concise and follow the standard syntax of adventure games.
        4. Try different verbs or nouns if necessary.
        5. Do not use quotes, periods, or exclamation marks in the command.
        6. Do not include explanations or reasoning in your response.
        
        REFINED COMMAND:
        """
        
        return prompt
    
    def _extract_action_from_response(self, response: str) -> str:
        """
        Extracts the action from the text generated by the LLM.
        
        Args:
            response: Full text generated by the LLM
            
        Returns:
            Extracted action
        """
        # Clean and format the response
        action = response.strip()
        
        # Remove common prefixes that the LLM might generate
        prefixes = [
            "The command is:",
            "Command:",
            "Action:",
            "Refined command:",
            ">",
            "$"
        ]
        
        for prefix in prefixes:
            if action.startswith(prefix):
                action = action[len(prefix):].strip()
        
        # Remove quotes if any
        action = action.strip('"\'')
        
        # Remove final punctuation
        action = re.sub(r'[.!?]$', '', action).strip()
        
        # Convert to lowercase for consistency
        action = action.lower()
        
        return action
    
    def _add_to_history(self, task: str, action: str, is_refinement: bool = False) -> None:
        """
        Adds an action to the history.
        
        Args:
            task: Task for which the action was generated
            action: Generated action
            is_refinement: Indicates if it is a refinement of a previous action
        """
        self.action_history.append((task, action))
        
        # Limit the size of the history (keep the last 100 actions)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]