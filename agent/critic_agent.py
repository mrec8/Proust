"""
Critic Agent for self-verification of task success.
"""
import logging
from typing import Dict, List, Tuple, Any, Optional

from utils.llm_interface import LLMInterface

class CriticAgent:
    """
    Agent that evaluates task success and provides feedback.
    """
    
    def __init__(self, config: Dict[str, Any], game_config: Dict[str, Any], 
                llm: LLMInterface):
        """
        Initializes the critic agent.
        
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
        
        # Critic agent parameters
        critic_config = self.config.get('critic_agent', {})
        self.strictness = critic_config.get('strictness', 0.8)
        
        # Evaluation history
        self.evaluation_history = []
        
        self.logger.info(f"Critic agent initialized with game: {self.game_name}")
    
    def check_task_success(self, task: str, agent_state: Dict[str, Any], 
                          action_history: List[str]) -> Tuple[bool, str]:
        """
        Verifies if a task has been successfully completed.
        
        Args:
            task: Task to evaluate
            agent_state: Current state of the agent
            action_history: List of actions taken for the task
            
        Returns:
            Tuple (success, critique)
        """
        # Build verification prompt
        prompt = self._build_verification_prompt(task, agent_state, action_history)
        
        # Generate evaluation
        response = self.llm.generate(prompt, temperature=0.3)
        
        # Extract evaluation result
        success, critique = self._extract_evaluation_result(response)
        
        # Log evaluation
        self._add_to_history(task, action_history, success, critique)
        
        self.logger.info(f"Task '{task}' evaluation: Success={success}")
        if not success and critique:
            self.logger.info(f"Critique: {critique}")
        
        return success, critique
    
    def _build_verification_prompt(self, task: str, agent_state: Dict[str, Any], 
                                  action_history: List[str]) -> str:
        """
        Builds the prompt to verify the success of a task.
        
        Args:
            task: Task to evaluate
            agent_state: Current state of the agent
            action_history: List of actions taken for the task
            
        Returns:
            Prompt for the LLM
        """
        # Extract relevant information from the state
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')
        
        # Format action history
        actions_text = ""
        for i, action in enumerate(action_history):
            actions_text += f"{i+1}. {action}\n"
        
        # Build the prompt
        prompt = f"""
        You are a Critic Agent evaluating whether a player successfully completed a task in a text-based interactive fiction game (e.g., Zork). You will receive the task, player actions, final environment output, and current inventory.

        Be flexible and contextual. A task may be completed through indirect or differently worded commands. Consider the **final game state**, not just the last message.

        TASK:
        {task}

        ACTIONS TAKEN:
        {actions_text}

        FINAL GAME STATE:
        {observation}

        INVENTORY:
        {inventory}

        --- SUCCESS CRITERIA BY CATEGORY ---

        1. EXAMINATION TASKS (e.g. "Examine mailbox"):
        ✅ Success: Game reveals any specific description or new info
        ❌ Failure: "You can't see any such thing", "Nothing special" without details

        2. ACQUISITION TASKS (e.g. "Take leaflet"):
        ✅ Success: "Taken." or item appears in inventory
        ❌ Failure: Still not in inventory or blocked message

        3. MOVEMENT TASKS (e.g. "Go north"):
        Movement tasks are special.
        ✅ Success: If moving in that direction is not possible. If location repeats after moving in that direction.
        IF THE GAME ALLOWS MOVEMENT, WHICH MEANS THE LOCATION/OBSERVATION IS NOW DIFFERENT, THEN:
            -With a random chance of 0.6 you consider the task to be not completed yet. Therefore FAILURE. So we advance again in that direction.
            In this case the critique should tell the player to try the same again.
            - With a random chance of 0.4 you consider the task to be completed. Therefore SUCCESS.
        4. OPENING/CLOSING TASKS (e.g. "Open mailbox"):
        ✅ Success: Game confirms it is open, *even if already open*
        ❌ Failure: Still closed or invalid object

        5. READING TASKS:
        ✅ Success: Any readable content shown
        ❌ Failure: "You can't read that", or generic rejection

        --- EVALUATION FORMAT ---

        Return your evaluation in this format:
        Success: true | false  
        Reasoning: Concise explanation grounded in the game’s actual response  
        Critique: [only if Success=false] Give a specific, actionable suggestion (e.g., "Try 'look under mat'" or "Use 'go east' instead of 'enter window'")

        Be forgiving with phrasing. If the task was effectively completed, mark it as success.

        --- EXAMPLES ---

        Task: "Open mailbox"  
        Game says: "That's already open."  
        ✅ → Success: true  
        Reasoning: The mailbox is confirmed open. Objective achieved.

        Task: "Enter house"  
        Action: "go east"  
        Response: "You are in the kitchen."  
        ✅ → Success: true  
        Reasoning: Player entered the house via alternate phrasing. Goal achieved.

        Task: "Look at mat"  
        Response: "Welcome to Zork!"  
        ❌ → Success: false  
        Critique: Try "look under mat" or "move mat" to trigger a more informative response.
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
        # Default values
        success = False
        critique = ""
        
        # Extract success indicator
        success_line = None
        for line in response.split('\n'):
            if line.lower().startswith('success:'):
                success_line = line.lower().replace('success:', '').strip()
                break
        
        if success_line:
            if 'true' in success_line:
                success = True
            elif 'false' in success_line:
                success = False
        
        # Extract critique if present
        critique_text = ""
        capture_critique = False
        for line in response.split('\n'):
            if line.lower().startswith('critique:'):
                capture_critique = True
                critique_text = line.replace('critique:', '').strip()
            elif capture_critique and line.strip():
                critique_text += ' ' + line.strip()
        
        if critique_text:
            critique = critique_text
        
        return success, critique
    
    def _add_to_history(self, task: str, action_history: List[str], 
                       success: bool, critique: str) -> None:
        """
        Adds an evaluation to the history.
        
        Args:
            task: Evaluated task
            action_history: Actions taken for the task
            success: Evaluation result
            critique: Provided critique
        """
        self.evaluation_history.append({
            'task': task,
            'actions': action_history,
            'success': success,
            'critique': critique
        })
        
        # Limit the size of the history (keep the last 50 evaluations)
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]
    
    def get_task_success_rate(self) -> float:
        """
        Calculates the success rate of evaluated tasks.
        
        Returns:
            Success rate as a float between 0 and 1
        """
        if not self.evaluation_history:
            return 0.0
        
        successful_count = sum(1 for eval_entry in self.evaluation_history 
                             if eval_entry['success'])
        return successful_count / len(self.evaluation_history)
    
    def get_common_failure_reasons(self, limit: int = 5) -> List[str]:
        """
        Identifies common reasons for task failures.
        
        Args:
            limit: Maximum number of reasons to return
            
        Returns:
            List of common failure reasons
        """
        if not self.evaluation_history:
            return []
        
        # Collect critiques from failed tasks
        critiques = [eval_entry['critique'] for eval_entry in self.evaluation_history 
                    if not eval_entry['success'] and eval_entry['critique']]
        
        # This is a simplified approach - in a real implementation, 
        # you might want to use clustering or more sophisticated NLP
        # to identify truly common patterns
        
        # For now, just return the most recent critiques
        return critiques[:limit]