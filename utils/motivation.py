"""
Motivation module to provide autonomy to the agent.
"""
import os
import json
import yaml
import time
import random
from typing import Dict, List, Any, Optional, Tuple

from utils.llm_interface import LLMInterface
from utils.memory import Memory

class MotivationModule:
    """
    Module that manages the agent's intrinsic motivation and autonomy.
    """
    
    def __init__(self, config_path: str, llm: LLMInterface, memory: Memory):
        """
        Initializes the motivation module.
        
        Args:
            config_path: Path to the configuration file
            llm: Interface with the language model
            memory: Agent's memory system
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm = llm
        self.memory = memory
        
        # Initialize motivation components
        self.curiosity = 0.8  # Initial curiosity level (0-1)
        self.mastery = 0.2    # Initial mastery level (0-1)
        self.autonomy = 0.5   # Initial autonomy level (0-1)
        
        # Emerging role
        self.role = None
        self.role_confidence = 0.0
        self.potential_roles = []
        
        # Value system
        self.values = {
            'exploration': 0.8,     # Preference for exploring vs. exploiting
            'risk_taking': 0.5,     # Preference for taking risks
            'social': 0.6,          # Preference for social interaction
            'achievement': 0.7,     # Preference for achievements and progress
            'collection': 0.5       # Preference for collecting objects
        }
        
        # Load previous state if it exists
        self._load_state()
    
    def update_motivation(self, task_result: Dict[str, Any]) -> None:
        """
        Updates motivation levels based on task results.
        
        Args:
            task_result: Information about the task result
        """
        task = task_result.get('task', '')
        success = task_result.get('success', False)
        result = task_result.get('result', '')
        
        # Update curiosity (decreases with repeated successes, increases with failures)
        if success:
            # Find similar tasks in memory
            similar_tasks = self._find_similar_tasks(task)
            
            # If we have already succeeded in similar tasks, curiosity decreases
            if similar_tasks:
                self.curiosity = max(0.2, self.curiosity - 0.05)
            else:
                # New type of success, slightly increases curiosity
                self.curiosity = min(1.0, self.curiosity + 0.02)
        else:
            # Failures increase curiosity (we want to understand why we failed)
            self.curiosity = min(1.0, self.curiosity + 0.05)
        
        # Update mastery (increases with successes, decreases with failures)
        if success:
            self.mastery = min(1.0, self.mastery + 0.03)
        else:
            self.mastery = max(0.1, self.mastery - 0.02)
        
        # Update autonomy (varies according to results)
        if success:
            self.autonomy = min(1.0, self.autonomy + 0.02)
        else:
            # If we fail but it is something new, maintain autonomy
            if 'never seen before' in result.lower() or 'unknown' in result.lower():
                pass  # Maintain current level
            else:
                self.autonomy = max(0.3, self.autonomy - 0.01)
        
        # Update value system based on the task
        self._update_values(task, success, result)
        
        # Update emerging role
        self._update_role(task, success, result)
        
        # Save state
        self._save_state()
    
    def generate_intrinsic_goal(self, agent_state: Dict[str, Any]) -> str:
        """
        Generates an intrinsic goal based on the current motivation.
        
        Args:
            agent_state: Current state of the agent
            
        Returns:
            Generated intrinsic goal
        """
        # Extract relevant information
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')
        score = agent_state.get('score', 0)
        
        # Build prompt to generate goal
        role_context = f"Your emerging role is: {self.role}" if self.role else "You do not have a defined role yet, you are in an exploration phase."
        
        prompt = f"""
        You are an autonomous agent in an interactive fiction game. Generate an intrinsic goal based on your current motivation and state.
        
        YOUR MOTIVATIONAL STATE:
        Curiosity: {self.curiosity:.2f} (Higher value = more interest in the unknown)
        Mastery: {self.mastery:.2f} (Higher value = more confidence in your abilities)
        Autonomy: {self.autonomy:.2f} (Higher value = greater self-initiative)
        
        YOUR EMERGING ROLE:
        {role_context}
        
        YOUR VALUES:
        Exploration: {self.values['exploration']:.2f}
        Risk-taking: {self.values['risk_taking']:.2f}
        Social interaction: {self.values['social']:.2f}
        Achievements: {self.values['achievement']:.2f}
        Collection: {self.values['collection']:.2f}
        
        YOUR CURRENT SITUATION:
        Observation: "{observation}"
        Inventory: "{inventory}"
        Score: {score}
        
        INSTRUCTIONS:
        1. Generate ONE intrinsic goal that reflects your current motivation and aligns with your emerging role.
        2. The goal must be specific, achievable, and internally motivated (not by an external reward).
        3. The goal must align with your personal values.
        4. Use a first-person format, such as "I want to explore..." or "I wish to find...".
        
        INTRINSIC GOAL:
        """
        
        # Generate goal
        response = self.llm.generate(prompt, temperature=0.7, max_tokens=100)
        
        # Clean and format the response
        goal = response.strip()
        
        # Limit the length
        if len(goal.split()) > 20:
            goal = ' '.join(goal.split()[:20])
        
        return goal
    
    def get_motivational_state(self) -> Dict[str, Any]:
        """
        Gets the current motivational state.
        
        Returns:
            Dictionary with the motivational state
        """
        return {
            'curiosity': self.curiosity,
            'mastery': self.mastery,
            'autonomy': self.autonomy,
            'role': self.role,
            'role_confidence': self.role_confidence,
            'potential_roles': self.potential_roles,
            'values': self.values
        }
    
    def get_role_description(self) -> str:
        """
        Generates a description of the emerging role.
        
        Returns:
            Description of the emerging role
        """
        if not self.role or self.role_confidence < 0.3:
            return "You are still exploring and discovering your place in this world. You have not adopted a specific role."
        
        # Build prompt for role description
        prompt = f"""
        Describe the role "{self.role}" for a character in an interactive fiction game.
        
        The character has the following values:
        Exploration: {self.values['exploration']:.2f}
        Risk-taking: {self.values['risk_taking']:.2f}
        Social interaction: {self.values['social']:.2f}
        Achievements: {self.values['achievement']:.2f}
        Collection: {self.values['collection']:.2f}
        
        INSTRUCTIONS:
        1. Describe this role in 2-3 sentences, using first person.
        2. Include main motivations and characteristic abilities.
        3. Mention how this role sees and interacts with the world.
        
        ROLE DESCRIPTION:
        """
        
        # Generate description
        response = self.llm.generate(prompt, temperature=0.6, max_tokens=150)
        
        return response.strip()
    
    def _find_similar_tasks(self, task: str) -> List[Dict[str, Any]]:
        """
        Finds similar tasks in episodic memory.
        
        Args:
            task: Task to compare
            
        Returns:
            List of similar tasks
        """
        return self.memory.get_relevant_memories(task, top_k=3)
    
    def _update_values(self, task: str, success: bool, result: str) -> None:
        """
        Updates the value system based on the completed task.
        
        Args:
            task: Task performed
            success: Whether the task was successful
            result: Result of the task
        """
        # Update exploration value
        if 'explore' in task.lower() or 'discover' in task.lower():
            if success:
                self.values['exploration'] = min(1.0, self.values['exploration'] + 0.02)
        
        # Update risk-taking value
        if 'dangerous' in task.lower() or 'risk' in task.lower() or 'attack' in task.lower():
            if success:
                self.values['risk_taking'] = min(1.0, self.values['risk_taking'] + 0.02)
            else:
                self.values['risk_taking'] = max(0.1, self.values['risk_taking'] - 0.01)
        
        # Update social value
        if 'talk' in task.lower() or 'ask' in task.lower() or 'conversation' in task.lower():
            if success:
                self.values['social'] = min(1.0, self.values['social'] + 0.02)
        
        # Update achievement value
        if success:
            self.values['achievement'] = min(1.0, self.values['achievement'] + 0.01)
        
        # Update collection value
        if 'collect' in task.lower() or 'obtain' in task.lower() or 'take' in task.lower():
            if success:
                self.values['collection'] = min(1.0, self.values['collection'] + 0.02)
    
    def _update_role(self, task: str, success: bool, result: str) -> None:
        """
        Updates the emerging role based on the completed task.
        
        Args:
            task: Task performed
            success: Whether the task was successful
            result: Result of the task
        """
        # If we do not have a defined role yet or confidence is low, try to infer one
        if not self.role or self.role_confidence < 0.5:
            # Build context for role inference
            episodes = self.memory.episodic_memory[-20:]  # Last 20 episodes
            
            episodes_summary = ""
            for i, episode in enumerate(episodes):
                episodes_summary += f"{i+1}. Task: {episode['task']}, Success: {episode['success']}\n"
            
            prompt = f"""
            Analyze the task history of a character in an interactive fiction game to infer their emerging role.
            
            TASK HISTORY:
            {episodes_summary}
            
            CHARACTER VALUES:
            Exploration: {self.values['exploration']:.2f}
            Risk-taking: {self.values['risk_taking']:.2f}
            Social interaction: {self.values['social']:.2f}
            Achievements: {self.values['achievement']:.2f}
            Collection: {self.values['collection']:.2f}
            
            INSTRUCTIONS:
            1. Based on the history and values, infer the most likely role for this character.
            2. The role should be a noun or short noun phrase (e.g., "Explorer", "Treasure Hunter", "Diplomat").
            3. Provide a brief justification.
            4. Assign a confidence between 0.0 and 1.0 to your inference.
            
            RESPONSE IN JSON FORMAT:
            {
                "role": "role name",
                "justification": "brief justification",
                "confidence": numeric_value
            }
            """
            
            # Generate role inference
            response = self.llm.generate(prompt, temperature=0.4, max_tokens=200)
            
            try:
                # Extract JSON
                import re
                json_match = re.search(r'{.*}', response, re.DOTALL)
                if json_match:
                    role_info = json.loads(json_match.group(0))
                    
                    inferred_role = role_info.get('role', '')
                    confidence = float(role_info.get('confidence', 0.3))
                    
                    # Update potential roles
                    if inferred_role and inferred_role not in [r['role'] for r in self.potential_roles]:
                        self.potential_roles.append({
                            'role': inferred_role,
                            'confidence': confidence,
                            'timestamp': time.time()
                        })
                    
                    # If confidence is sufficient, update the role
                    if confidence > self.role_confidence:
                        self.role = inferred_role
                        self.role_confidence = confidence
            except:
                pass  # If there is an error, maintain the current role
        else:
            # If we already have an established role, reinforce it if the task is consistent
            role_lower = self.role.lower()
            task_lower = task.lower()
            
            # Check if the task reinforces the current role
            reinforces_role = False
            
            # Examples of role reinforcements
            if 'explorer' in role_lower or 'adventurer' in role_lower:
                if 'explore' in task_lower or 'discover' in task_lower:
                    reinforces_role = True
            
            elif 'warrior' in role_lower or 'hunter' in role_lower:
                if 'attack' in task_lower or 'fight' in task_lower or 'kill' in task_lower:
                    reinforces_role = True
            
            elif 'scholar' in role_lower or 'researcher' in role_lower:
                if 'read' in task_lower or 'study' in task_lower or 'examine' in task_lower:
                    reinforces_role = True
            
            # If the task reinforces the role and is successful, increase confidence
            if reinforces_role and success:
                self.role_confidence = min(1.0, self.role_confidence + 0.05)
            # If the task contradicts the role, slightly decrease confidence
            elif not reinforces_role:
                self.role_confidence = max(0.3, self.role_confidence - 0.01)
    
    def _save_state(self) -> None:
        """Saves the motivational state to disk."""
        save_dir = 'logs/motivation'
        os.makedirs(save_dir, exist_ok=True)
        
        state = {
            'curiosity': self.curiosity,
            'mastery': self.mastery,
            'autonomy': self.autonomy,
            'role': self.role,
            'role_confidence': self.role_confidence,
            'potential_roles': self.potential_roles,
            'values': self.values,
            'timestamp': time.time()
        }
        
        with open(os.path.join(save_dir, 'motivation_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self) -> None:
        """Loads the motivational state from disk."""
        state_path = 'logs/motivation/motivation_state.json'
        
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.curiosity = state.get('curiosity', self.curiosity)
            self.mastery = state.get('mastery', self.mastery)
            self.autonomy = state.get('autonomy', self.autonomy)
            self.role = state.get('role', self.role)
            self.role_confidence = state.get('role_confidence', self.role_confidence)
            self.potential_roles = state.get('potential_roles', self.potential_roles)
            self.values = state.get('values', self.values)