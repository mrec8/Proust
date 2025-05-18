"""
Skill Manager for storing and retrieving skills.
"""
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np

from utils.llm_interface import LLMInterface

class Skill:
    """A class representing a skill in text adventure games."""
    
    def __init__(self, description: str, task: str, commands: List[str], 
                result: str, contexts: Optional[List[str]] = None):
        """
        Initialize a skill.
        
        Args:
            description: Description of what the skill does
            task: The task this skill helps accomplish
            commands: List of commands that implement the skill
            result: The result of executing the commands
            contexts: List of contexts where this skill is applicable
        """
        self.description = description
        self.task = task
        self.commands = commands
        self.result = result
        self.contexts = contexts or []
        self.created_at = time.time()
        self.used_count = 0
        self.success_count = 0
    
    def use(self, success: bool = True) -> None:
        """
        Record a use of this skill.
        
        Args:
            success: Whether the use was successful
        """
        self.used_count += 1
        if success:
            self.success_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary for serialization."""
        return {
            'description': self.description,
            'task': self.task,
            'commands': self.commands,
            'result': self.result,
            'contexts': self.contexts,
            'created_at': self.created_at,
            'used_count': self.used_count,
            'success_count': self.success_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Skill':
        """Create skill from dictionary."""
        skill = cls(
            description=data['description'],
            task=data['task'],
            commands=data['commands'],
            result=data['result'],
            contexts=data.get('contexts', [])
        )
        skill.created_at = data.get('created_at', time.time())
        skill.used_count = data.get('used_count', 0)
        skill.success_count = data.get('success_count', 0)
        return skill


class SkillManager:
    """Manager for storing and retrieving skills."""
    
    def __init__(self, config: Dict[str, Any], game_name:str, llm: LLMInterface):
        """
        Initialize the skill manager.
        
        Args:
            config: Configuration dictionary
            llm: Language model interface for generating skill descriptions
        """
        self.config = config
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Storage for skills
        self.skills = []
        
        # Directory to save/load skills
        self.save_dir = self.config.get('skill_manager', {}).get('save_dir', 'skills')
        self.save_dir = os.path.join(self.save_dir, game_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load existing skills if available
        self._load_skills()
        
        self.logger.info(f"Skill manager initialized with {len(self.skills)} skills")
    
    def add_skill(self, task: str, commands: List[str], result: str, success: bool) -> Optional[Skill]:
        """
        Add a new skill to the library.
        
        Args:
            task: The task the skill accomplishes
            commands: The commands used to complete the task
            result: The result of executing the commands
            success: Whether the task was successfully completed
            
        Returns:
            The added skill or None if not successful
        """
        if not success or not commands:
            return None
        
        # Generate a description for the skill
        description = self._generate_skill_description(task, commands, result)
        
        # Create contexts from the task and commands
        contexts = [task] + [cmd for cmd in commands]
        
        # Create the skill
        skill = Skill(description, task, commands, result, contexts)
        
        # Check for duplicate skills
        if not self._is_duplicate_skill(skill):
            self.skills.append(skill)
            self.logger.info(f"Added new skill: {description}")
            
            # Save the updated skills
            self._save_skills()
            
            return skill
        
        self.logger.info(f"Skill similar to '{description}' already exists")
        return None
    
    def retrieve_skills(self, task: str, agent_state: Dict[str, Any], limit: int = 5) -> List[Skill]:
        """
        Retrieve relevant skills for a task.
        
        Args:
            task: The task to find skills for
            agent_state: Current state of the agent
            limit: Maximum number of skills to return
            
        Returns:
            List of relevant skills
        """
        if not self.skills:
            return []
        
        # Simple retrieval based on keyword matching
        scores = []
        for skill in self.skills:
            score = self._calculate_relevance_score(skill, task, agent_state)
            scores.append(score)
        
        # Get the indices of the top-scoring skills
        if not scores:
            return []
        
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = indices[:limit]
        
        # Return the top skills
        top_skills = [self.skills[i] for i in top_indices]
        
        self.logger.info(f"Retrieved {len(top_skills)} skills for task: {task}")
        return top_skills
    

    def _generate_skill_description(self, task: str, commands: List[str], result: str) -> str:
        """
        Generate a description for a skill.
        
        Args:
            task: The task the skill accomplishes
            commands: The commands used
            result: The result of executing the commands
            
        Returns:
            A description of the skill
        """
        # Extract the key command - usually the first or last command is most relevant
        key_command = commands[-1] if commands else "unknown"
        
        # Build prompt for skill description
        prompt = f"""
        You are a professional text adventure game analyst creating precise skill descriptions 
        for an intelligent agent. Your task is to create an extremely concise, action-oriented 
        description that captures exactly what this skill accomplishes.

        TASK PERFORMED:
        {task}

        COMMANDS USED:
        {', '.join(commands)}

        RESULT ACHIEVED:
        {result}

        SKILL DESCRIPTION SPECIFICATIONS:
        1. FORMAT REQUIREMENTS:
        - Must start with a present-tense action verb
        - Must be EXACTLY 4-8 words total
        - Must express a complete thought
        - Must focus on the OUTCOME, not the attempt
        
        2. CONTENT REQUIREMENTS:
        - Describe SPECIFICALLY what was accomplished
        - Include the relevant object or location
        - Avoid meta-language ("completes task", "succeeds at")
        - Avoid vague terms ("checks", "interacts with", "tries to")
        
        3. CLARITY REQUIREMENTS:
        - Must be immediately understandable without context
        - Must distinguish this skill from similar skills
        - Must be concrete, not abstract
        - Must be factual, not aspirational

        EXAMPLES OF EXCELLENT SKILL DESCRIPTIONS:
        - "Opens mailbox revealing small leaflet" (not "Checks mailbox")
        - "Takes leaflet from open mailbox" (not "Gets item")
        - "Navigates north to forest clearing" (not "Explores forest")
        - "Reads welcome message on leaflet" (not "Examines paper")
        - "Unlocks door with brass key" (not "Uses key")
        - "Crafts torch using rope and stick" (not "Makes light source")

        EXAMPLES OF POOR SKILL DESCRIPTIONS (NEVER USE THESE PATTERNS):
        - "Checking for leaflet availability" (too vague, not outcome-focused)
        - "Successfully opens the container" (uses meta-language, not specific)
        - "Trying to find the key" (describes attempt, not outcome)
        - "Getting item from mailbox" (not specific enough)
        - "Going to another location" (not specific enough)
        - "Completing the first puzzle successfully" (uses meta-language)

        ANALYSIS PROCESS:
        1. Identify the EXACT outcome that was achieved
        2. Identify the specific objects or locations involved
        3. Select the most precise action verb
        4. Construct a concise phrase with these elements
        5. Verify it meets ALL requirements above

        YOUR RESPONSE MUST BE EXACTLY ONE PHRASE OF 4-8 WORDS THAT MEETS ALL REQUIREMENTS ABOVE.
        DO NOT include explanations, alternatives, or commentary.

        SKILL DESCRIPTION:
        """
        
        # Generate description
        response = self.llm.generate(prompt, temperature=0.4, max_tokens=50)
        
        # Clean and format the description
        description = response.strip()
        
        # Limit length
        if len(description.split()) > 15:
            description = ' '.join(description.split()[:15])
            if not description.endswith('.'):
                description += '.'
        
        return description
    
    def _calculate_relevance_score(self, skill: Skill, task: str, agent_state: Dict[str, Any]) -> float:
        """
        Calculate relevance score for a skill.
        
        Args:
            skill: The skill to evaluate
            task: The current task
            agent_state: Current state of the agent
            
        Returns:
            Relevance score (higher is more relevant)
        """
        score = 0.0
        
        # Task similarity (most important)
        task_lower = task.lower()
        if skill.task.lower() == task_lower:
            score += 10.0
        else:
            # Check for keyword overlap
            task_keywords = set(task_lower.split())
            skill_task_keywords = set(skill.task.lower().split())
            overlap = task_keywords.intersection(skill_task_keywords)
            score += len(overlap) * 2.0
        
        # Context relevance
        observation = agent_state.get('observation', '').lower()
        inventory = agent_state.get('inventory', '').lower()
        
        # Check if context keywords appear in current state
        for context in skill.contexts:
            context_lower = context.lower()
            context_keywords = context_lower.split()
            
            for keyword in context_keywords:
                if len(keyword) < 3:
                    continue  # Skip very short words
                    
                if keyword in observation:
                    score += 0.5
                if keyword in inventory:
                    score += 0.5
        
        # Success rate bonus
        if skill.used_count > 0:
            success_rate = skill.success_count / skill.used_count
            score += success_rate * 1.0
        
        return score
    
    def _is_duplicate_skill(self, new_skill: Skill) -> bool:
        """
        Check if a skill is a duplicate of an existing skill.
        
        Args:
            new_skill: The skill to check
            
        Returns:
            True if the skill is a duplicate
        """
        for skill in self.skills:
            # If task and commands are identical, it's a duplicate
            if skill.task == new_skill.task and set(skill.commands) == set(new_skill.commands):
                return True
            
            # If description is very similar, it might be a duplicate
            if skill.description.lower() == new_skill.description.lower():
                return True
        
        return False
    
    def _save_skills(self) -> None:
        """Save skills to disk."""
        skills_data = [skill.to_dict() for skill in self.skills]
        filepath = os.path.join(self.save_dir, 'skills.json')
        
        try:
            with open(filepath, 'w') as f:
                json.dump(skills_data, f, indent=2)
            self.logger.info(f"Saved {len(skills_data)} skills to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving skills: {e}")
    
    def _load_skills(self) -> None:
        """Load skills from disk."""
        filepath = os.path.join(self.save_dir, 'skills.json')
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    skills_data = json.load(f)
                
                self.skills = [Skill.from_dict(data) for data in skills_data]
                self.logger.info(f"Loaded {len(self.skills)} skills from {filepath}")
            except Exception as e:
                self.logger.error(f"Error loading skills: {e}")
                self.skills = []
        else:
            self.logger.info("No saved skills found. Starting with empty skill library.")
            self.skills = []
    
    def get_skill_count(self) -> int:
        """Get the number of skills in the library."""
        return len(self.skills)
    
    def get_skill_statistics(self) -> Dict[str, Any]:
        """Get statistics about the skill library."""
        if not self.skills:
            return {
                "count": 0,
                "avg_commands_per_skill": 0,
                "avg_success_rate": 0,
                "top_skills": []
            }
        
        # Calculate statistics
        avg_commands = sum(len(skill.commands) for skill in self.skills) / len(self.skills)
        
        success_rates = []
        for skill in self.skills:
            if skill.used_count > 0:
                success_rates.append(skill.success_count / skill.used_count)
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Get top skills by usage
        top_skills = sorted(self.skills, key=lambda s: s.used_count, reverse=True)[:5]
        top_skill_info = [
            {
                "description": skill.description,
                "used_count": skill.used_count,
                "success_rate": skill.success_count / skill.used_count if skill.used_count > 0 else 0
            }
            for skill in top_skills
        ]
        
        return {
            "count": len(self.skills),
            "avg_commands_per_skill": avg_commands,
            "avg_success_rate": avg_success_rate,
            "top_skills": top_skill_info
        }