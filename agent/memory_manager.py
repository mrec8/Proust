"""
Memory Manager for storing and retrieving memory.
"""
import os
import json
import logging
import random
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from utils.llm_interface import LLMInterface

class Memory:
    """A memory about the game world."""
    
    def __init__(self, topic: str, observation: str, inference: str, 
                contexts: Optional[List[str]] = None):
        """
        Initialize a memory.
        
        Args:
            topic: Main topic or subject of this memory
            observation: What was directly observed
            inference: What can be deduced or inferred from this observation
            contexts: List of contexts where this memory is relevant
        """
        self.topic = topic
        self.observation = observation
        self.inference = inference
        self.contexts = contexts or []
        self.created_at = time.time()
        self.retrieved_count = 0
        self.relevance_score = 0.0  # How useful this memory has been
    
    def retrieve(self) -> None:
        """Record that this memory has been retrieved."""
        self.retrieved_count += 1
    
    def get_topic(self) -> str:
        """Get the topic of this memory."""
        return self.topic

    def update_relevance(self, score: float) -> None:
        """Update the relevance score of this memory."""
        self.relevance_score = 0.8 * self.relevance_score + 0.2 * score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            'topic': self.topic,
            'observation': self.observation,
            'inference': self.inference,
            'contexts': self.contexts,
            'created_at': self.created_at,
            'retrieved_count': self.retrieved_count,
            'relevance_score': self.relevance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary."""
        memory = cls(
            topic=data['topic'],
            observation=data['observation'],
            inference=data['inference'],
            contexts=data.get('contexts', [])
        )
        memory.created_at = data.get('created_at', time.time())
        memory.retrieved_count = data.get('retrieved_count', 0)
        memory.relevance_score = data.get('relevance_score', 0.0)
        return memory

class MemoryManager:
    """Manager for storing and retrieving memories about the game world."""
    
    def __init__(self, config: Dict[str, Any], llm: LLMInterface, game_name: str):
        """
        Initialize the memory manager.
        
        Args:
            config: Configuration dictionary
            llm: Language model interface
        """
        self.config = config
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Storage for memories
        self.memories = []
        
        # Directory to save/load memories
        self.save_dir = self.config.get('memory_manager', {}).get('save_dir', 'memories')
        self.save_dir = os.path.join(self.save_dir, game_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load existing memories if available
        self._load_memories()
        
        self.logger.info(f"Memory manager initialized with {len(self.memories)} memories")
    
    def add_memory(self, observation: str, agent_state: Dict[str, Any]) -> Optional[Memory]:
        """
        Analyze an observation and potentially add a new memory.
        
        Args:
            observation: Environment observation
            agent_state: Current state of the agent
            
        Returns:
            The added memory or None if none was added
        """
        # Generate a memory based on the observation
        memory_data = self._generate_memory(observation, agent_state)
        
        if not memory_data:
            return None
        
        topic, observation_text, inference = memory_data
        
        # Refine the memory through reflection
        refined_memory = self._reflect_on_memory(topic, observation_text, inference, agent_state)
        
        if refined_memory:
            topic, observation_text, inference = refined_memory
        
        # Create contexts from the observation and state
        contexts = [topic]
        location = agent_state.get('location', '')
        if location:
            contexts.append(f"location:{location}")
        
        # Create the memory
        memory = Memory(topic, observation_text, inference, contexts)
        
        # Check if it's a duplicate memory
        if not self._is_duplicate_memory(memory):
            self.memories.append(memory)
            self.logger.info(f"Added new memory: {topic}")
            
            # Save the updated memories
            self._save_memories()
            
            return memory
        
        self.logger.info(f"Memory similar to '{topic}' already exists")
        return None

    def _reflect_on_memory(self, topic: str, observation: str, inference: str, 
                        agent_state: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        """
        Reflect on and potentially improve a generated memory.
        
        Args:
            topic: Memory topic
            observation: Memory observation
            inference: Memory inference
            agent_state: Current agent state
            
        Returns:
            Refined memory tuple or None if no improvement
        """
        # Build prompt for reflection
        prompt = f"""
        You are an expert game analyst reviewing a memory that has been generated about a text adventure game.
        
        ORIGINAL MEMORY:
        TOPIC: {topic}
        OBSERVATION: {observation}
        INFERENCE: {inference}
        
        The memory is about to be stored in the agent's memory system for future reference.
        
        Your task is to evaluate and improve this memory to make it more valuable for future decision-making.
        Consider the following aspects:
        
        1. TOPIC - Is it specific and descriptive enough? Does it capture the key insight?
        2. OBSERVATION - Does it contain the most relevant part of the original text?
        3. INFERENCE - Is the deduction valuable, accurate, and actionable for future gameplay?
        
        If the memory is already excellent, respond with "MEMORY_APPROVED".
        
        If you can improve the memory, provide your revised version in the following format:
        REFINED_TOPIC: [improved topic]
        REFINED_OBSERVATION: [improved observation]
        REFINED_INFERENCE: [improved inference]
        
        Focus on making the memory more:
        - Precise (captures the exact information needed)
        - Actionable (useful for future decisions)
        - Concise (removes unnecessary details)
        - Insightful (extracts non-obvious implications)
        
        YOUR RESPONSE:
        """
        
        # Generate reflection
        response = self.llm.generate(prompt, temperature=0.3, max_tokens=200)
        response = response.strip()
        
        # Check if any refinement is suggested
        if "MEMORY_APPROVED" in response:
            return None  # No changes needed
        
        # Extract refined components
        refined_topic = topic
        refined_observation = observation
        refined_inference = inference
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("REFINED_TOPIC:"):
                refined_topic = line[14:].strip()
            elif line.startswith("REFINED_OBSERVATION:"):
                refined_observation = line[20:].strip()
            elif line.startswith("REFINED_INFERENCE:"):
                refined_inference = line[18:].strip()
        
        # Return the refined memory if there were actually changes
        if (refined_topic != topic or 
            refined_observation != observation or 
            refined_inference != inference):
            return (refined_topic, refined_observation, refined_inference)
        
        return None  # No meaningful changes
    
    def retrieve_memories(self, current_context: str, agent_state: Dict[str, Any], max_limit: int = 5) -> List[Memory]:
        """
        Retrieve relevant memories for a context.
        
        Args:
            current_context: The current context (task, observation, etc.)
            agent_state: Current state of the agent
            max_limit: Maximum number of memories to return
            
        Returns:
            List of relevant memories
        """
        if not self.memories:
            return []
        
        # Calculate relevance scores for all memories
        scores = []
        for memory in self.memories:
            score = self._calculate_relevance_score(memory, current_context, agent_state)
            scores.append(score)
        
        # Create memory-score pairs
        memory_scores = list(zip(self.memories, scores))
        
        # Sort by score (highest first)
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter memories with a relevance threshold
        # Only include memories with meaningful relevance (score > 1.0)
        relevant_memories = [m for m, s in memory_scores if s > 1.0]
        
        # If we have no relevant memories after filtering, return an empty list
        if not relevant_memories:
            self.logger.info("No sufficiently relevant memories found for the current context")
            return []
            
        # Cap at max_limit if needed
        if len(relevant_memories) > max_limit:
            relevant_memories = relevant_memories[:max_limit]
        
        # Update retrieval count for used memories
        for memory in relevant_memories:
            memory.retrieve()
        
        self.logger.info(f"Retrieved {len(relevant_memories)} relevant memories for context: {current_context[:50]}...")
        
        # Log the scores of retrieved memories
        for i, memory in enumerate(relevant_memories):
            score = memory_scores[i][1]
            self.logger.debug(f"Memory: '{memory.topic}' - Relevance score: {score:.2f}")
        
        return relevant_memories
    
    def _generate_memory(self, observation: str, agent_state: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        """
        Generate a memory based on an observation.
        
        Args:
            observation: Environment observation
            agent_state: Current state of the agent
            
        Returns:
            Tuple (topic, observation, inference) or None if nothing memorable
        """
        # Build prompt for memory generation
        prompt = self._build_memory_generation_prompt(observation, agent_state)
        
        # Generate memory
        response = self.llm.generate(prompt, temperature=0.4, max_tokens=200)
        
        # Parse the response
        return self._parse_memory_response(response)
    
    def _build_memory_generation_prompt(self, observation: str, agent_state: Dict[str, Any]) -> str:
        """Build the prompt for generating a memory."""
        prompt = """
        You are an intelligent agent exploring a text adventure game. Analyze the following observation 
        from the game and extract valuable information that would be worth remembering for future interactions.
        Not formal things regarding the game, but rather useful information that can help in decision-making. Content oriented.
        
        CURRENT OBSERVATION:
        {observation}
        
        CURRENT INVENTORY:
        {inventory}
        
        Your task is to identify if there is anything worth remembering from this observation. Focus on information that:
        1. Helps relate aspects of the game world
        2. Provides clues about puzzles or obstacles
        3. Describes important objects and their properties or uses
        4. Reveals character behaviors or weaknesses
        5. Identifies environmental hazards or features
        6. Contains location details that might be important for navigation
        7. Describes successful interactions that could be repeated elsewhere
        8. DOES NOT include generic observations or standard game messages or information about the FORMAL STRUCTURE of the game.
        If there is nothing particularly memorable in this observation, respond with "NOTHING_MEMORABLE".
        
        If there IS something worth remembering, format your response as:
        TOPIC: [A short 2-5 word title for this memory]
        OBSERVATION: [The specific part of the text that contains the valuable information]
        INFERENCE: [What we can infer or deduce from this observation that might be useful later]
        
        GUIDELINES FOR GOOD MEMORIES:
        - Focus only on the MOST important insight - do not extract multiple memories
        - The TOPIC should be concise and descriptive of the key discovery
        - The OBSERVATION should be a direct quote or paraphrase from the text
        - The INFERENCE should explain why this information is useful and how it might apply in the future
        - Focus on unique, specific information - avoid generic observations
        - Prioritize actionable information that affects gameplay decisions
        
        YOUR RESPONSE:
        """
        
        # Fill in the placeholders
        prompt = prompt.replace("{observation}", observation)
        prompt = prompt.replace("{inventory}", agent_state.get('inventory', 'No inventory information available'))
        
        return prompt
    
    def _parse_memory_response(self, response: str) -> Optional[Tuple[str, str, str]]:
        """Parse the LLM response to extract memory components."""
        response = response.strip()
        
        # Check if nothing memorable
        if "NOTHING_MEMORABLE" in response:
            return None
        
        # Extract topic, observation and inference
        topic = ""
        observation = ""
        inference = ""
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("TOPIC:"):
                topic = line[6:].strip()
            elif line.startswith("OBSERVATION:"):
                observation = line[12:].strip()
            elif line.startswith("INFERENCE:"):
                inference = line[10:].strip()
        
        # Check that we have all components
        if topic and observation and inference:
            return (topic, observation, inference)
        
        return None
    
    def _calculate_relevance_score(self, memory: Memory, current_context: str, agent_state: Dict[str, Any]) -> float:
        """
        Calculate relevance score for a memory.
        
        Args:
            memory: The memory to evaluate
            current_context: The current context (task + observation)
            agent_state: Current state of the agent
            
        Returns:
            Relevance score (higher is more relevant)
        """
        score = 0.0
        
        # Parse the current context
        current_task = ""
        current_observation = ""
        
        if "Task:" in current_context:
            current_task = current_context.split("Task:")[1].split("\n")[0].strip()
        
        if "Observation:" in current_context:
            current_observation = current_context.split("Observation:")[1].split("\n")[0].strip()
        else:
            current_observation = agent_state.get('observation', '')
        
        # Extract current location and inventory information
        current_inventory = agent_state.get('inventory', '').lower()
        
        # Score boosts for direct topic matches with task
        if current_task:
            task_lower = current_task.lower()
            
            # 1. Direct topic match with task (highest importance)
            if memory.topic.lower() in task_lower:
                score += 5.0
            elif any(word in task_lower for word in memory.topic.lower().split() if len(word) > 3):
                score += 3.0
        
        # 2. Topic or observation keywords in current observation
        if current_observation:
            obs_lower = current_observation.lower()
            
            # Check if any significant words from the memory topic appear in the current observation
            topic_words = [word for word in memory.topic.lower().split() if len(word) > 3]
            for word in topic_words:
                if word in obs_lower:
                    score += 2.0
                    break
            
            # Check if objects from memory observation appear in current observation
            memory_obj_words = [word for word in memory.observation.lower().split() if len(word) > 3]
            obj_matches = sum(1 for word in memory_obj_words if word in obs_lower)
            score += min(obj_matches * 0.5, 2.0)  # Cap at 2.0
        
        # 3. Memory mentions objects that are in current inventory
        inventory_words = [word for word in current_inventory.split() if len(word) > 3]
        for word in inventory_words:
            if word in memory.observation.lower():
                score += 1.5
                break
        
        # 4. Memory references similar actions to current task
        task_verbs = ["take", "get", "drop", "examine", "look", "open", "close", "put", "attack"]
        memory_verbs = []
        for verb in task_verbs:
            if verb in memory.topic.lower() or verb in memory.inference.lower():
                memory_verbs.append(verb)
        
        task_verb_matches = sum(1 for verb in memory_verbs if verb in current_task.lower())
        score += task_verb_matches * 1.0
        
        # 5. Memory has been useful before (usage history bonus)
        if memory.retrieved_count > 0:
            usage_bonus = min(memory.retrieved_count * 0.3, 2.0)  # Cap at 2.0
            score += usage_bonus
        
        # 6. Recency factor (less important than relevance)
        age_in_hours = (time.time() - memory.created_at) / 3600
        recency_factor = max(0.8, 1.0 - (age_in_hours / 240))  # Decay over 10 days, but never below 0.8
        score *= recency_factor
        
        # Apply memory's own relevance score if it exists
        if hasattr(memory, 'relevance_score') and memory.relevance_score > 0:
            score *= (1.0 + memory.relevance_score * 0.5)
        
        return score
    
    def _is_duplicate_memory(self, new_memory: Memory) -> bool:
        """
        Check if a memory is a duplicate of an existing memory.
        
        Args:
            new_memory: The memory to check
            
        Returns:
            True if the memory is a duplicate
        """
        for memory in self.memories:
            # If topic and observation are identical, it's a duplicate
            if memory.topic == new_memory.topic and memory.observation == new_memory.observation:
                return True
            
            # If the observation is very similar, it might be a duplicate
            observation_similarity = self._calculate_text_similarity(
                memory.observation.lower(), 
                new_memory.observation.lower()
            )
            
            if observation_similarity > 0.8:  # High similarity threshold
                return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate a simple similarity score between two texts."""
        # This is a simple implementation - in a real system, you might use
        # more sophisticated text similarity measures
        
        # Split into words and create sets
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _save_memories(self) -> None:
        """Save memories to disk."""
        memories_data = [memory.to_dict() for memory in self.memories]
        filepath = os.path.join(self.save_dir, 'memories.json')
        
        try:
            with open(filepath, 'w') as f:
                json.dump(memories_data, f, indent=2)
            self.logger.info(f"Saved {len(memories_data)} memories to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving memories: {e}")
    
    def _load_memories(self) -> None:
        """Load memories from disk."""
        filepath = os.path.join(self.save_dir, 'memories.json')
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    memories_data = json.load(f)
                
                self.memories = [Memory.from_dict(data) for data in memories_data]
                self.logger.info(f"Loaded {len(self.memories)} memories from {filepath}")
            except Exception as e:
                self.logger.error(f"Error loading memories: {e}")
                self.memories = []
        else:
            self.logger.info("No saved memories found. Starting with empty memory library.")
            self.memories = []



    def should_create_memory(self, observation: str, agent_state: Dict[str, Any], 
                            previous_observation: Optional[str] = None) -> bool:
        """
        Determines if a memory should be created from the current observation.
        
        Args:
            observation: Current observation
            agent_state: Current state of the agent
            previous_observation: Previous observation for comparison
            
        Returns:
            Boolean indicating if a memory should be created
        """
        # Don't create memories for very short observations (likely just transition messages)
        if len(observation.split()) < 5:
            return False
            
        # Don't create memories for standard messages
        standard_responses = [
            "You can't go that way", 
            "I don't understand that",
            "You can't see any such thing",
            "Nothing happens",
            "That's not a verb I recognize"
        ]
        
        for response in standard_responses:
            if response.lower() in observation.lower():
                return False
        
        # Check if this is a new location (high value for memory creation)
        is_new_location = False
        if previous_observation:
            # Simple heuristic: if the first line changed significantly, it might be a new location
            prev_first_line = previous_observation.split("\n")[0] if "\n" in previous_observation else previous_observation
            curr_first_line = observation.split("\n")[0] if "\n" in observation else observation
            
            if prev_first_line != curr_first_line and len(curr_first_line.split()) > 3:
                is_new_location = True
        
        if is_new_location:
            return True
            
        # Check if the observation contains interesting keywords
        interesting_keywords = [
            "discover", "notice", "realize", "find", "see", "hidden", "secret", 
            "unusual", "strange", "mysterious", "warning", "danger", "important",
            "key", "door", "locked", "open", "close", "reveals", "appears",
            "message", "note", "letter", "inscription", "writing"
        ]
        
        for keyword in interesting_keywords:
            if keyword.lower() in observation.lower():
                return True
        
        # Check if there was a significant change in inventory
        # This would require tracking previous inventory state
        
        # If nothing above triggered, use a probabilistic approach
        # Create memories less frequently for common observations
        if len(self.memories) < 10:  # If we have few memories, be more generous
            return random.random() < 0.3  # 30% chance
        else:
            return random.random() < 0.1  # 10% chance