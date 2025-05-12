"""
Automatic Curriculum Agent to propose tasks adapted to progress.
"""
import logging
from typing import Dict, List, Any, Optional
import yaml

from utils.llm_interface import LLMInterface

class CurriculumAgent:
    """
    Agent that proposes tasks adapted to the agent's progress and narrative context.
    """
    
    def __init__(self, config: Dict[str, Any], game_config: Dict[str, Any], 
                llm: LLMInterface):
        """
        Initializes the curriculum agent.
        
        Args:
            config: General configuration dictionary
            game_config: Game-specific configuration dictionary
            llm: Interface with the language model
        """
        self.config = config
        self.game_config = game_config
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Specific configurations
        self.game_name = self.config['environment']['game']
        self.game_specific_config = self.game_config.get(self.game_name, {})
        
        self.special_commands = self.game_specific_config.get('special_commands', [])

        # Initialize task history
        self.completed_tasks = []
        self.failed_tasks = []
        self.current_task = None
        
        # Difficulty parameters
        self.difficulty_scaling = self.config.get('curriculum_agent', {}).get('task_difficulty_scaling', 1.2)
        self.max_failed_tasks = self.config.get('curriculum_agent', {}).get('max_failed_tasks_memory', 10)
        
        # Store the initial task if available
        self.initial_goal = self.game_specific_config.get('starting_goal', 
                                                         "Explore the world and discover its mechanics")
        
        self.logger.info(f"Curriculum agent initialized with game: {self.game_name}")
    
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
        
        

        # Extra check to prevent repeating failed tasks
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            # Generate the next task using the LLM
            response = self.llm.generate(prompt, temperature=0.7)
            
            # Extract the task from the generated text
            task = self._extract_task_from_response(response)
            # Check if this is a repeat of a recently failed task
            if not self._is_similar_to_failed_task(task):
                break
                
            attempts += 1
            self.logger.info(f"Rejected task '{task}' as similar to a recently failed task. Attempt {attempts}/{max_attempts}")
        
        # Ensure task is concise
        task = self._format_task(task)
        
        # Update the current task
        self.current_task = task
        self.logger.info(f"Proposed new task: {task}")
        
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
            In the text adventure game '{self.game_name}', the general objective is: "{self.initial_goal}".
            
            As a first specific and concrete task to start exploring this game, the agent should:
            """
            response = self.llm.generate(prompt, temperature=0.5, max_tokens=50)
            initial_task = response.strip()
            
            # If the response is too generic, use a default task
            if len(initial_task.split()) < 3:
                # Game-specific defaults
                if "zork" in self.game_name.lower():
                    return "Look around the current location and examine visible objects"
                else:
                    return "Explore the initial location and examine the environment"
            
            self.logger.info(f"Generated initial task: {initial_task}")
            return initial_task
        
        # If there is no initial goal, generate a basic initial task
        return "Look around and examine your surroundings"
    
    def _get_exploration_progress(self) -> Dict[str, Any]:
        """
        Calculates exploration progress metrics based on completed and failed tasks.
        
        Returns:
            Dictionary with progress metrics
        """
        # Count tasks by category
        task_categories = {
            "exploration": 0,  # Exploration of places
            "inventory": 0,    # Managing inventory
            "interaction": 0,  # Interaction with objects
            "puzzle": 0,       # Solving puzzles
            "conversation": 0  # Conversation with NPCs
        }
        
        # Classify completed tasks
        for task in self.completed_tasks:
            task_lower = task.lower()
            if any(word in task_lower for word in ["explore", "go", "move", "look", "search"]):
                task_categories["exploration"] += 1
            elif any(word in task_lower for word in ["get", "take", "drop", "inventory"]):
                task_categories["inventory"] += 1
            elif any(word in task_lower for word in ["use", "open", "close", "push", "pull"]):
                task_categories["interaction"] += 1
            elif any(word in task_lower for word in ["solve", "unlock", "find", "figure"]):
                task_categories["puzzle"] += 1
            elif any(word in task_lower for word in ["talk", "ask", "tell", "answer"]):
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
        
        # Extract progress information
        completed_count = exploration_progress["completed_tasks_count"]
        failed_count = exploration_progress["failed_tasks_count"]
        success_rate = exploration_progress["success_rate"]
        task_categories = exploration_progress["task_categories"]
        recent_completed = exploration_progress["completed_tasks"]
        recent_failed = exploration_progress["failed_tasks"]
        
        # Build the prompt
        prompt = f"""
        You are an intelligent curriculum agent for the text adventure game '{self.game_name}'.
        Your task is to propose the next immediate objective for an agent to complete.

        CURRENT GAME STATE:
        Current observation: "{observation}"
        Current inventory: "{inventory}"

        EXPLORATION PROGRESS:
        Tasks completed so far: {completed_count}
        Tasks failed so far: {failed_count}
        Success rate: {success_rate:.2f}

        Recently completed tasks:
        {', '.join(recent_completed) if recent_completed else 'None'}

        Recently failed tasks:
        {', '.join(recent_failed) if recent_failed else 'None'}

        GAME INFORMATION:
        Special commands: {', '.join(self.special_commands)}
        Key locations: {', '.join(self.game_specific_config.get('key_locations', []))}
        Key objects: {', '.join(self.game_specific_config.get('key_objects', []))}

        TASK CRITERIA:
        1. The task MUST be extremely specific and verifiable (e.g., "Open the mailbox", "Go north")
        2. The task MUST be achievable with a few simple commands
        3. The task MUST be simple for beginners if the success rate is low
        4. The task MUST be a single action or a simple sequence 
        5. The task MUST NOT repeat ANY recently failed task - DO NOT propose the same task again
        6. The task MUST be expressed in 5-10 words maximum
        7. The task MUST be very different from failed tasks - try a completely new direction

        Your response must be ONLY the task itself (e.g., "Examine the mailbox", "Go north", "Take the leaflet").
        DO NOT include explanations, steps, or any additional text.

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
    
    def _is_similar_to_failed_task(self, task: str) -> bool:
        """Check if the proposed task is similar to a recently failed task."""
        if not self.failed_tasks:
            return False
            
        task_lower = task.lower()
        
        # Check the most recent failed tasks (last 5)
        recent_failures = self.failed_tasks[-5:]
        
        for failed_task in recent_failures:
            failed_lower = failed_task.lower()
            
            # Direct match
            if task_lower == failed_lower:
                return True
                
            # Significant word overlap
            task_words = set(task_lower.split())
            failed_words = set(failed_lower.split())
            
            if len(task_words.intersection(failed_words)) >= 2:  # If 2+ shared words
                return True
        
        return False

    def _format_task(self, task: str) -> str:
        """Format the task to be concise and clean."""
        # Remove common prefixes
        prefixes = [
            "The next task is:",
            "Next task:",
            "Task:",
            "The task is:"
        ]
        
        for prefix in prefixes:
            if task.startswith(prefix):
                task = task[len(prefix):].strip()
        
        # Truncate to the first sentence if multiple exist
        if '.' in task:
            sentences = task.split('.')
            task = sentences[0].strip() + '.'
            
        # Ensure reasonable length (max 10 words)
        words = task.split()
        if len(words) > 10:
            task = ' '.join(words[:10])
            if not task.endswith(('.', '!', '?')):
                task += '.'
                
        return task.strip()
    
    def add_completed_task(self, task: str) -> None:
        """
        Registers a task as completed.
        
        Args:
            task: Description of the completed task
        """
        # Add to the list of completed tasks
        self.completed_tasks.append(task)
        self.logger.info(f"Task completed: {task}")
        
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
        self.logger.info(f"Task failed: {task}")
        
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