"""
Automatic Curriculum Agent to propose tasks adapted to progress.
"""
import logging
from typing import Dict, List, Any, Optional
import yaml

from agent.memory_manager import Memory
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
    
    def propose_next_task(self, agent_state: Dict[str, Any], 
                          observation_history: List[str],
                          memories: Optional[List[Memory]] = None) -> str:
        """
        Proposes the next task based on the current state, progress, and memories.
        
        Args:
            agent_state: Current state of the agent
            memories: List of relevant memories to consider
            
        Returns:
            Description of the next task to perform
        """
        # If it is the first task, use the initial goal
        if not self.completed_tasks and not self.failed_tasks:
            self.current_task = self._generate_initial_task()
            return self.current_task
        
        # Get relevant information to generate the next task
        exploration_progress = self._get_exploration_progress()
        
        # Build prompt for the LLM, now including memories
        prompt = self._build_task_proposal_prompt(agent_state, exploration_progress, observation_history, memories)
        
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
        return "Look around"
    
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
                                    observation_history: List[str],
                                    exploration_progress: Dict[str, Any],
                                    memories: Optional[List[Memory]] = None) -> str:
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')

        # Completed and failed tasks context
        task_context = ""
        if self.completed_tasks:
            task_context += "COMPLETED TASKS:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(self.completed_tasks)]) + "\n"
        if self.failed_tasks:
            task_context += "FAILED TASKS:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(self.failed_tasks)]) + "\n"

        # Object interaction context
        interacted_objects = set()
        for task in self.completed_tasks + self.failed_tasks:
            for obj in self.game_specific_config.get('key_objects', []):
                if obj.lower() in task.lower():
                    interacted_objects.add(obj.lower())

      

                
        # Memory context
        memory_context = ""
        if memories:
            memory_context += "MEMORIES:\n"
            for i, m in enumerate(memories):
                memory_context += (
                    f"{i+1}. Topic: {m.topic}\n"
                    f"   Observation: {m.observation}\n"
                    f"   Inference: {m.inference}\n"
                )

        # -----------------------------
        # Final prompt to return
        # -----------------------------
        return f"""You are a curriculum task planner embedded inside an autonomous interactive fiction agent playing the classic game Zork. Your role is to carefully propose the next most promising high-level task for the agent to pursue, taking into account its current sensory input, recent actions, inventory, and task history.

        ## CONTEXT AVAILABLE TO YOU:

        You will be provided with:
        - `observation`: A textual description of the current game location, including visible objects and exits.
        - `inventory`: The current items the agent is carrying.
        - `previous_tasks`: A chronological list of proposed tasks, each annotated with success (True/False).
        - `observation_history`: The most recent environment observations made by the agent.
        - `steps_without_change`: How many consecutive steps have occurred without a meaningful change in location, objects, or environment description.

        ## YOUR OBJECTIVES:

        1. Propose **a single, clear, high-level task** that is:
        - Achievable based on the current `observation`.
        - Different from tasks that have recently failed.
        - Not redundant with previous successes or overly similar phrasing.
        - Focused on **progression**, such as acquiring new objects, interacting meaningfully with the environment, or uncovering new descriptions.

        2. **Avoid suggesting movement/navigation** (e.g., "go north") if:
        - That direction was already tried recently and led to a loop or no location change.
        - The `steps_without_change` counter is high, suggesting exploration is stuck.
        - No visible exit or landmark suggests value in that direction.

        3. **Avoid loops and non-progressive commands**:
        - Repetitive commands like “look” or “wait” should only be proposed if clearly justified by the current state (e.g., a new object just appeared).
        - Do not suggest examining or opening items that have already been interacted with unless a new reason is present.

        4. Prefer tasks that:
        - Directly interact with objects mentioned in the `observation`.
        - Continue the logical chain of interaction (e.g., after opening a box, "take what’s inside", then "read it").
        - Help the agent uncover more of the game world or story.

        ## FORMAT:
        Output **only** the next best high-level task as a natural language imperative (e.g., `"Open the mailbox"`, `"Read the leaflet"`). Do not add explanations, summaries, or code.

        ---

        # DATA FOR YOU TO USE:

        observation: {observation}  
        inventory: {inventory}  
        previous_tasks: {task_context}  
        observation_history: {observation_history}  
        

        Now propose the most meaningful next task for the agent."""



    
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