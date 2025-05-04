"""
Memory system for the agent.
"""
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class Memory:
    """
    Memory system for storing the agent's experiences and knowledge.
    """
    
    def __init__(self, save_dir: str = 'logs/memory'):
        """
        Initializes the memory system.
        
        Args:
            save_dir: Directory to save the memory
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize memory components
        self.episodic_memory = []  # Episodic memory (experiences)
        self.semantic_memory = {}  # Semantic memory (knowledge)
        self.entity_memory = {}    # Entity memory (objects, characters, locations)
        
        # Load existing memory if any
        self._load_memory()
    
    def add_episode(self, task: str, actions: List[str], result: str, 
                   agent_state: Dict[str, Any], success: bool) -> None:
        """
        Adds an episode to the episodic memory.
        
        Args:
            task: Task performed
            actions: List of actions taken
            result: Result of the actions
            agent_state: Agent's state at the end of the episode
            success: Whether the task was successful
        """
        episode = {
            'task': task,
            'actions': actions,
            'result': result,
            'state': {
                'observation': agent_state.get('observation', ''),
                'inventory': agent_state.get('inventory', ''),
                'score': agent_state.get('score', 0),
                'moves': agent_state.get('moves', 0)
            },
            'success': success,
            'timestamp': time.time()
        }
        
        self.episodic_memory.append(episode)
        
        # Extract knowledge from the episode
        self._extract_knowledge(episode)
        
        # Save memory
        self._save_memory()
    
    def add_knowledge(self, concept: str, knowledge: str, source: str = 'inference') -> None:
        """
        Adds knowledge to the semantic memory.
        
        Args:
            concept: Concept or topic of the knowledge
            knowledge: Description of the knowledge
            source: Source of the knowledge
        """
        # Normalize the concept (lowercase)
        concept_key = concept.lower()
        
        # Create or update knowledge entry
        if concept_key not in self.semantic_memory:
            self.semantic_memory[concept_key] = []
        
        # Add the new knowledge
        knowledge_entry = {
            'description': knowledge,
            'source': source,
            'timestamp': time.time(),
            'confidence': 0.8  # Default value, could be adjusted
        }
        
        # Avoid duplicates
        for existing in self.semantic_memory[concept_key]:
            if existing['description'] == knowledge:
                # Update confidence and timestamp if it already exists
                existing['confidence'] = max(existing['confidence'], 0.8)
                existing['timestamp'] = time.time()
                return
        
        self.semantic_memory[concept_key].append(knowledge_entry)
        
        # Save memory
        self._save_memory()
    
    def add_entity(self, entity_type: str, name: str, properties: Dict[str, Any]) -> None:
        """
        Adds an entity to the entity memory.
        
        Args:
            entity_type: Type of entity (object, character, location)
            name: Name of the entity
            properties: Properties of the entity
        """
        # Normalize the name (lowercase)
        entity_key = name.lower()
        
        # Create structure if it doesn't exist
        if entity_type not in self.entity_memory:
            self.entity_memory[entity_type] = {}
        
        # Create or update entity
        if entity_key not in self.entity_memory[entity_type]:
            self.entity_memory[entity_type][entity_key] = {
                'name': name,
                'properties': properties,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'sightings': 1
            }
        else:
            # Update existing entity
            entity = self.entity_memory[entity_type][entity_key]
            entity['last_seen'] = time.time()
            entity['sightings'] += 1
            
            # Update properties (keep existing and add new ones)
            for key, value in properties.items():
                entity['properties'][key] = value
        
        # Save memory
        self._save_memory()
    
    def get_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memories for a query.
        
        Args:
            query: Query text
            top_k: Number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        # For a simple implementation, we search for keyword matches
        # In a more advanced implementation, semantic embeddings would be used
        
        # Normalize the query
        query_words = set(query.lower().split())
        
        # Calculate relevance for each episode
        scored_episodes = []
        for episode in self.episodic_memory:
            score = 0
            
            # Search in the task
            task_words = set(episode['task'].lower().split())
            score += len(query_words.intersection(task_words)) * 2  # Higher weight for tasks
            
            # Search in the result
            result_words = set(episode['result'].lower().split())
            score += len(query_words.intersection(result_words))
            
            # Search in the actions
            for action in episode['actions']:
                action_words = set(action.lower().split())
                score += len(query_words.intersection(action_words))
            
            if score > 0:
                scored_episodes.append((episode, score))
        
        # Sort by relevance
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top_k most relevant
        relevant_episodes = [episode for episode, _ in scored_episodes[:top_k]]
        
        return relevant_episodes
    
    def get_knowledge(self, concept: str) -> List[Dict[str, Any]]:
        """
        Retrieves knowledge about a concept.
        
        Args:
            concept: Concept to search for
            
        Returns:
            List of knowledge about the concept
        """
        # Normalize the concept
        concept_key = concept.lower()
        
        # Search for exact knowledge
        if concept_key in self.semantic_memory:
            return self.semantic_memory[concept_key]
        
        # Search for partial knowledge
        partial_matches = []
        for key, knowledge_list in self.semantic_memory.items():
            # If the concept is contained in the key or vice versa
            if concept_key in key or key in concept_key:
                # Add each knowledge with a reference to its original concept
                for knowledge in knowledge_list:
                    partial_match = knowledge.copy()
                    partial_match['original_concept'] = key
                    partial_matches.append(partial_match)
        
        return partial_matches
    
    def get_entity(self, entity_type: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific entity.
        
        Args:
            entity_type: Type of entity
            name: Name of the entity
            
        Returns:
            Entity information or None if not found
        """
        # Normalize the name
        entity_key = name.lower()
        
        # Check if it exists
        if entity_type in self.entity_memory and entity_key in self.entity_memory[entity_type]:
            return self.entity_memory[entity_type][entity_key]
        
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Retrieves all entities of a specific type.
        
        Args:
            entity_type: Type of entity
            
        Returns:
            List of entities of the specified type
        """
        if entity_type in self.entity_memory:
            return list(self.entity_memory[entity_type].values())
        
        return []
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generates a summary of the current memory state.
        
        Returns:
            Memory summary
        """
        # Count episodes by result
        successful_episodes = sum(1 for e in self.episodic_memory if e['success'])
        failed_episodes = len(self.episodic_memory) - successful_episodes
        
        # Count entities by type
        entity_counts = {entity_type: len(entities) 
                        for entity_type, entities in self.entity_memory.items()}
        
        # Count concepts in semantic memory
        knowledge_count = len(self.semantic_memory)
        
        # Calculate time statistics
        current_time = time.time()
        if self.episodic_memory:
            first_episode_time = min(e['timestamp'] for e in self.episodic_memory)
            elapsed_time = (current_time - first_episode_time) / 3600  # Hours
        else:
            elapsed_time = 0
        
        return {
            'episodic_memory': {
                'total': len(self.episodic_memory),
                'successful': successful_episodes,
                'failed': failed_episodes
            },
            'semantic_memory': {
                'concepts': knowledge_count,
            },
            'entity_memory': entity_counts,
            'elapsed_time_hours': elapsed_time
        }
    
    def _extract_knowledge(self, episode: Dict[str, Any]) -> None:
        """
        Extracts knowledge and entity information from an episode.
        
        Args:
            episode: Episode to extract knowledge from
        """
        # Simple example: extract objects from the inventory as entities
        if 'inventory' in episode['state']:
            inventory_text = episode['state']['inventory']
            
            # Use a simple implementation to detect objects
            # In a real implementation, more advanced NLP would be used
            common_objects = [
                "sword", "key", "lantern", "book", "scroll", "food", "water",
                "map", "compass", "torch", "letter", "note", "ring", "gem"
            ]
            
            for obj in common_objects:
                if obj in inventory_text.lower():
                    self.add_entity('object', obj, {
                        'description': f"An object found in inventory",
                        'portable': True,
                        'seen_in_episode': episode['task']
                    })
        
        # Extract locations from the observation
        if 'observation' in episode['state']:
            observation = episode['state']['observation']
            
            # Search for patterns like "You are in [location]"
            location_patterns = [
                r"You are in (.*?)\.",
                r"You are standing (.*?)\.",
                r"You're in (.*?)\."
            ]
            
            import re
            for pattern in location_patterns:
                match = re.search(pattern, observation)
                if match:
                    location = match.group(1).strip()
                    self.add_entity('location', location, {
                        'description': f"A location in the game",
                        'seen_in_episode': episode['task']
                    })
    
    def _save_memory(self) -> None:
        """Saves the memory to disk."""
        # Save episodic memory
        episodic_path = os.path.join(self.save_dir, 'episodic_memory.json')
        with open(episodic_path, 'w') as f:
            json.dump(self.episodic_memory, f, indent=2)
        
        # Save semantic memory
        semantic_path = os.path.join(self.save_dir, 'semantic_memory.json')
        with open(semantic_path, 'w') as f:
            json.dump(self.semantic_memory, f, indent=2)
        
        # Save entity memory
        entity_path = os.path.join(self.save_dir, 'entity_memory.json')
        with open(entity_path, 'w') as f:
            json.dump(self.entity_memory, f, indent=2)
    
    def _load_memory(self) -> None:
        """Loads the memory from disk."""
        # Load episodic memory
        episodic_path = os.path.join(self.save_dir, 'episodic_memory.json')
        if os.path.exists(episodic_path):
            with open(episodic_path, 'r') as f:
                self.episodic_memory = json.load(f)
        
        # Load semantic memory
        semantic_path = os.path.join(self.save_dir, 'semantic_memory.json')
        if os.path.exists(semantic_path):
            with open(semantic_path, 'r') as f:
                self.semantic_memory = json.load(f)
        
        # Load entity memory
        entity_path = os.path.join(self.save_dir, 'entity_memory.json')
        if os.path.exists(entity_path):
            with open(entity_path, 'r') as f:
                self.entity_memory = json.load(f)