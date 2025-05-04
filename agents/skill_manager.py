"""
Skill Library Manager to store and retrieve skills.
"""
import os
import yaml
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import re

from utils.llm_interface import LLMInterface

class SkillManager:
    """
    Manager that handles the agent's skill library.
    """
    
    def __init__(self, config_path: str, llm: LLMInterface, embedding_model):
        """
        Initializes the skill library manager.
        
        Args:
            config_path: Path to the configuration file
            llm: Interface with the language model
            embedding_model: Model to generate embeddings
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm = llm
        self.embedding_model = embedding_model
        
        # Specific configurations
        self.embedding_dim = self.config['skill_library'].get('embedding_dim', 384)
        self.top_k = self.config['skill_library'].get('top_k_retrievals', 5)
        
        # Initialize the skill library (in-memory for now)
        self.skills = []
        self.skill_embeddings = []
        
        # Directory to save/load
        self.save_dir = os.path.join('skills', 'vector_db')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load existing skills if any
        self._load_skills()
    
    def add_skill(self, task: str, actions: List[str], result: str, was_successful: bool) -> None:
        """
        Adds a new skill to the library.
        
        Args:
            task: Description of the task
            actions: List of actions/commands used
            result: Result of executing the actions
            was_successful: Indicates if the skill successfully completed the task
        """
        if not was_successful:
            return  # Do not store skills that were not successful
        
        # Generate a description of the skill
        skill_description = self._generate_skill_description(task, actions, result)
        
        # Generate the embedding for the description
        embedding = self._generate_embedding(skill_description)
        
        # Create the skill object
        skill = {
            'task': task,
            'description': skill_description,
            'actions': actions,
            'result': result,
            'timestamp': time.time(),
            'uses': 0
        }
        
        # Add to the library
        self.skills.append(skill)
        self.skill_embeddings.append(embedding)
        
        # Save the updated library
        self._save_skills()
        
        return skill
    
    def retrieve_skills(self, task: str, agent_state: Dict[str, Any], 
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant skills for a given task.
        
        Args:
            task: Description of the task
            agent_state: Current state of the agent
            top_k: Number of skills to retrieve (uses default value if None)
            
        Returns:
            List of relevant skills
        """
        if not self.skills:
            return []
        
        # Use the configured top_k if not specified
        if top_k is None:
            top_k = self.top_k
        
        # Create an enriched query context
        query_context = self._build_query_context(task, agent_state)
        
        # Generate embedding for the query context
        query_embedding = self._generate_embedding(query_context)
        
        # Calculate similarities
        similarities = self._calculate_similarities(query_embedding)
        
        # Get the indices of the top_k most similar skills
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Collect the relevant skills
        relevant_skills = [self.skills[idx] for idx in top_indices]
        
        # Update the usage counter
        for idx in top_indices:
            self.skills[idx]['uses'] += 1
        
        # Save the updated library
        self._save_skills()
        
        return relevant_skills
    
    def _generate_skill_description(self, task: str, actions: List[str], result: str) -> str:
        """
        Generates a concise description of a skill.
        
        Args:
            task: Description of the task
            actions: List of actions/commands used
            result: Result of executing the actions
            
        Returns:
            Description of the skill
        """
        # Build the prompt
        prompt = f"""
        Generate a concise description of a skill in an interactive fiction game.
        
        TASK:
        {task}
        
        ACTIONS TAKEN:
        {', '.join(actions)}
        
        RESULT:
        {result}
        
        INSTRUCTIONS:
        1. Create a clear and concise description of this skill (1-2 sentences).
        2. The description should be useful for reusing this skill in similar situations.
        3. Do not include specific details like proper names or exact locations.
        4. Use a general format that describes the sequence of actions and their purpose.
        5. The description should start with an infinitive verb (e.g., "Open", "Obtain").
        
        SKILL DESCRIPTION:
        """
        
        # Generate the description
        response = self.llm.generate(prompt, temperature=0.4, max_tokens=100)
        
        # Clean the response
        description = response.strip()
        
        # Limit the length of the description
        if len(description.split()) > 20:
            words = description.split()
            description = ' '.join(words[:20])
        
        return description
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generates a vector embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Use the embedding model to generate the vector
        embedding = self.embedding_model.encode(text)
        
        # Normalize the embedding (optional but recommended for similarity comparisons)
        normalized_embedding = embedding / np.linalg.norm(embedding)
        
        return normalized_embedding
    
    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculates similarities between a query embedding and the library embeddings.
        
        Args:
            query_embedding: Query embedding
            
        Returns:
            Array of similarities
        """
        # Convert the list of embeddings to a numpy matrix
        skill_embeddings_matrix = np.array(self.skill_embeddings)
        
        # Calculate cosine similarity
        similarities = np.dot(skill_embeddings_matrix, query_embedding)
        
        return similarities
    
    def _build_query_context(self, task: str, agent_state: Dict[str, Any]) -> str:
        """
        Builds an enriched query context.
        
        Args:
            task: Description of the task
            agent_state: Current state of the agent
            
        Returns:
            Enriched query context
        """
        # Extract relevant information from the state
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')
        
        # Generate an enhanced query that considers the context
        prompt = f"""
        Given a task in an interactive fiction game, generate an enriched query
        that can be used to search for relevant skills in a library.
        
        TASK:
        {task}
        
        CURRENT OBSERVATION:
        {observation}
        
        INVENTORY:
        {inventory}
        
        INSTRUCTIONS:
        1. Analyze the task and context.
        2. Identify key objectives, actions, objects, and locations.
        3. Generate an expanded query that captures the essence of what needs to be done.
        4. Include synonyms or related terms to improve retrieval.
        5. Keep the query concise but informative (2-3 sentences).
        
        ENRICHED QUERY:
        """
        
        # Generate the enriched query
        response = self.llm.generate(prompt, temperature=0.4, max_tokens=150)
        
        # Clean the response
        query_context = response.strip()
        
        return f"{task} {query_context}"
    
    def _save_skills(self) -> None:
        """Saves the skill library to disk."""
        skills_path = os.path.join(self.save_dir, 'skills.json')
        embeddings_path = os.path.join(self.save_dir, 'embeddings.npy')
        
        # Save skills
        with open(skills_path, 'w') as f:
            json.dump(self.skills, f, indent=2)
        
        # Save embeddings
        if self.skill_embeddings:
            np.save(embeddings_path, np.array(self.skill_embeddings))
    
    def _load_skills(self) -> None:
        """Loads the skill library from disk."""
        skills_path = os.path.join(self.save_dir, 'skills.json')
        embeddings_path = os.path.join(self.save_dir, 'embeddings.npy')
        
        # Load skills if they exist
        if os.path.exists(skills_path):
            with open(skills_path, 'r') as f:
                self.skills = json.load(f)
        
        # Load embeddings if they exist
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            self.skill_embeddings = [embedding for embedding in embeddings]