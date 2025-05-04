"""
Automatic Curriculum Agent to propose tasks adapted to progress.
"""
import os
import yaml
import json
import random
from typing import Dict, List, Tuple, Set, Any, Optional

from utils.llm_interface import LLMInterface

class CurriculumAgent:
    """
    Agent that proposes tasks adapted to the agent's progress and narrative context.
    """
    
    def __init__(self, config_path: str, game_config_path: str, llm: LLMInterface):
        """
        Initializes the curriculum agent.
        
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
        
        # Initialize task history
        self.completed_tasks = []
        self.failed_tasks = []
        self.current_task = None
        
        # Difficulty parameters
        self.difficulty_scaling = self.config['agents']['curriculum_agent'].get('task_difficulty_scaling', 1.2)
        self.max_failed_tasks = self.config['agents']['curriculum_agent'].get('max_failed_tasks_memory', 10)
        
        # Initialize with game objectives if available
        self.progression_milestones = self.game_specific_config.get('progression_milestones', [])
        
        # Store the initial task if available
        self.initial_goal = self.game_specific_config.get('starting_goal', 
                                                         "Explore the world and discover its mechanics")
    
    def propose_next_task(self, agent_state: Dict[str, Any]) -> str:
        """
        Proposes the next task based on the current state and progress.
        
        Args:
            agent_state: Current state of the agent
            
        Returns:
            Description of the next task to perform
        """
        # If it is the first task, use the initial goal
        if not self.completed_tasks and not self.failed_tasks:
            self.current_task = self._generate_initial_task()
            return self.current_task
        
        # Get relevant information to generate the next task
        exploration_progress = self._get_exploration_progress()
        
        # Build prompt for the LLM
        prompt = self._build_task_proposal_prompt(agent_state, exploration_progress)
        
        # Generate the next task using the LLM
        response = self.llm.generate(prompt, temperature=0.7)
        
        # Extract the task from the generated text
        task = self._extract_task_from_response(response)
        
        # Update the current task
        self.current_task = task
        
        return task
    
    def _generate_initial_task(self) -> str:
        """
        Generates the first task for the agent.
        
        Returns:
            Description of the initial task
        """
        # If there is an initial goal in the configuration, use it
        if self.initial_goal:
            # Decompose the initial goal into a specific task
            prompt = f"""
            In the game '{self.game_name}', the general objective is: "{self.initial_goal}".
            
            As a first specific and concrete task to start, the agent should:
            """
            response = self.llm.generate(prompt, temperature=0.5, max_tokens=50)
            initial_task = response.strip()
            
            # If the response is too generic, use a default task
            if len(initial_task.split()) < 3:
                if self.game_name == "zork1":
                    return "Explore the surroundings of the white house"
                else:
                    return f"Explore the initial location and examine the environment"
            
            return initial_task
        
        # If there is no initial goal, generate a basic initial task
        return "Explore the surroundings and familiarize yourself with the environment"
    
    def _get_exploration_progress(self) -> Dict[str, Any]:
        """
        Calculates exploration progress metrics based on completed and failed tasks.
        
        Returns:
            Dictionary with progress metrics
        """
        # Count tasks by category
        task_categories = {
            "exploration": 0,  # Exploration of places
            "collection": 0,   # Collection of objects
            "interaction": 0,  # Interaction with objects
            "puzzle": 0,       # Solving puzzles
            "combat": 0,       # Combat with entities
            "conversation": 0  # Conversation with NPCs
        }
        
        # Classify completed tasks
        for task in self.completed_tasks:
            task_lower = task.lower()
            if any(word in task_lower for word in ["explore", "go", "visit", "find place"]):
                task_categories["exploration"] += 1
            elif any(word in task_lower for word in ["get", "collect", "take"]):
                task_categories["collection"] += 1
            elif any(word in task_lower for word in ["use", "open", "close", "move", "push", "pull"]):
                task_categories["interaction"] += 1
            elif any(word in task_lower for word in ["solve", "unlock", "decipher"]):
                task_categories["puzzle"] += 1
            elif any(word in task_lower for word in ["attack", "kill", "fight"]):
                task_categories["combat"] += 1
            elif any(word in task_lower for word in ["talk", "ask", "answer"]):
                task_categories["conversation"] += 1
        
        # Calculate success rate
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        success_rate = len(self.completed_tasks) / total_tasks if total_tasks > 0 else 0
        
        return {
            "completed_tasks_count": len(self.completed_tasks),
            "failed_tasks_count": len(self.failed_tasks),
            "success_rate": success_rate,
            "task_categories": task_categories,
            "completed_tasks": self.completed_tasks[-5:],  # Last 5 completed tasks
            "failed_tasks": self.failed_tasks[-5:]         # Last 5 failed tasks
        }
    
    def _build_task_proposal_prompt(self, agent_state: Dict[str, Any], 
                                   exploration_progress: Dict[str, Any]) -> str:
        """
        Builds the prompt to generate the next task.
        
        Args:
            agent_state: Current state of the agent
            exploration_progress: Exploration progress metrics
            
        Returns:
            Prompt for the LLM
        """
        # Extract relevant information from the state
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')
        score = agent_state.get('score', 0)
        moves = agent_state.get('moves', 0)
        
        # Extract progress information
        completed_count = exploration_progress["completed_tasks_count"]
        failed_count = exploration_progress["failed_tasks_count"]
        success_rate = exploration_progress["success_rate"]
        task_categories = exploration_progress["task_categories"]
        recent_completed = exploration_progress["completed_tasks"]
        recent_failed = exploration_progress["failed_tasks"]
        
        # Build the prompt
        prompt = f"""
        You are an intelligent curriculum agent for an interactive fiction game called {self.game_name}.
        Your task is to propose the next immediate objective for an agent exploring this narrative world.
        
        GAME INFORMATION:
        {self.game_specific_config.get('description', 'A text adventure game.')}
        
        CURRENT AGENT STATE:
        Current observation: "{observation}"
        Current inventory: "{inventory}"
        Current score: {score}
        Moves made: {moves}
        
        EXPLORATION PROGRESS:
        Tasks completed so far: {completed_count}
        Tasks failed so far: {failed_count}
        Success rate: {success_rate:.2f}
        
        Categories of completed tasks:
        - Exploration: {task_categories["exploration"]}
        - Collection: {task_categories["collection"]}
        - Interaction: {task_categories["interaction"]}
        - Puzzles: {task_categories["puzzle"]}
        - Combat: {task_categories["combat"]}
        - Conversation: {task_categories["conversation"]}
        
        Recently completed tasks:
        {', '.join(recent_completed) if recent_completed else 'None'}
        
        Recently failed tasks:
        {', '.join(recent_failed) if recent_failed else 'None'}
        
        CRITERIA FOR THE NEXT TASK:
        1. The task must be specific, concrete, and verifiable.
        2. The task must be achievable with the agent's current state and resources.
        3. The task must be adapted to the agent's current skill level.
        4. The task must contribute to exploration and progress in the game.
        5. The task must not exactly repeat something the agent just failed.
        6. The task must maintain narrative coherence with the game world.
        
        INSTRUCTIONS:
        Propose ONE specific task that the agent should attempt next.
        The task should be brief and start with an infinitive verb (e.g., "Explore", "Collect", "Open").
        Do not include explanations, justifications, or additional instructions.
        
        NEXT TASK:
        """
        
        return prompt
    
    def _extract_task_from_response(self, response: str) -> str:
        """
        Extracts the task from the text generated by the LLM.
        
        Args:
            response: Full text generated by the LLM
            
        Returns:
            Extracted task
        """
        # Clean and format the response
        task = response.strip()
        
        # Remove common prefixes that the LLM might generate
        prefixes = [
            "The next task is:",
            "Next task:",
            "Task:",
            "The task is:"
        ]
        
        for prefix in prefixes:
            if task.startswith(prefix):
                task = task[len(prefix):].strip()
        
        # Limit the length of the task to be concise
        if len(task.split()) > 10:
            # Take only the first 10 words
            words = task.split()
            task = ' '.join(words[:10])
            
            # Ensure it ends with a punctuation mark
            if not task.endswith(('.', '!', '?')):
                task += '.'
        
        return task
    
    def add_completed_task(self, task: str) -> None:
        """
        Registers a task as completed.
        
        Args:
            task: Description of the completed task
        """
        # Add to the list of completed tasks
        self.completed_tasks.append(task)
        
        # If it was the current task, clear it
        if self.current_task == task:
            self.current_task = None
    
    def add_failed_task(self, task: str) -> None:
        """
        Registers a task as failed.
        
        Args:
            task: Description of the failed task
        """
        # Add to the list of failed tasks
        self.failed_tasks.append(task)
        
        # Limit the number of remembered failed tasks
        if len(self.failed_tasks) > self.max_failed_tasks:
            self.failed_tasks = self.failed_tasks[-self.max_failed_tasks:]
        
        # If it was the current task, clear it
        if self.current_task == task:
            self.current_task = None
    
    def get_completed_tasks(self) -> List[str]:
        """
        Returns the list of completed tasks.
        
        Returns:
            List of completed tasks
        """
        return self.completed_tasks
    
    def get_failed_tasks(self) -> List[str]:
        """
        Returns the list of failed tasks.
        
        Returns:
            List of failed tasks
        """
        return self.failed_tasks