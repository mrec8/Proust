"""
Gestor de Biblioteca de Habilidades para almacenar y recuperar habilidades.
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
    Gestor que maneja la biblioteca de habilidades del agente.
    """
    
    def __init__(self, config_path: str, llm: LLMInterface, embedding_model):
        """
        Inicializa el gestor de biblioteca de habilidades.
        
        Args:
            config_path: Ruta al archivo de configuración
            llm: Interfaz con el modelo de lenguaje
            embedding_model: Modelo para generar embeddings
        """
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm = llm
        self.embedding_model = embedding_model
        
        # Configuraciones específicas
        self.embedding_dim = self.config['skill_library'].get('embedding_dim', 384)
        self.top_k = self.config['skill_library'].get('top_k_retrievals', 5)
        
        # Inicializar la biblioteca de habilidades (memoria in-memory por ahora)
        self.skills = []
        self.skill_embeddings = []
        
        # Directorio para guardar/cargar
        self.save_dir = os.path.join('skills', 'vector_db')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Cargar habilidades existentes si hay
        self._load_skills()
    
    def add_skill(self, task: str, actions: List[str], result: str, was_successful: bool) -> None:
        """
        Agrega una nueva habilidad a la biblioteca.
        
        Args:
            task: Descripción de la tarea
            actions: Lista de acciones/comandos utilizados
            result: Resultado de la ejecución de las acciones
            was_successful: Indica si la habilidad completó la tarea con éxito
        """
        if not was_successful:
            return  # No almacenar habilidades que no tuvieron éxito
        
        # Generar una descripción de la habilidad
        skill_description = self._generate_skill_description(task, actions, result)
        
        # Generar el embedding para la descripción
        embedding = self._generate_embedding(skill_description)
        
        # Crear el objeto de habilidad
        skill = {
            'task': task,
            'description': skill_description,
            'actions': actions,
            'result': result,
            'timestamp': time.time(),
            'uses': 0
        }
        
        # Agregar a la biblioteca
        self.skills.append(skill)
        self.skill_embeddings.append(embedding)
        
        # Guardar la biblioteca actualizada
        self._save_skills()
        
        return skill
    
    def retrieve_skills(self, task: str, agent_state: Dict[str, Any], 
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Recupera habilidades relevantes para una tarea dada.
        
        Args:
            task: Descripción de la tarea
            agent_state: Estado actual del agente
            top_k: Número de habilidades a recuperar (usa el valor predeterminado si None)
            
        Returns:
            Lista de habilidades relevantes
        """
        if not self.skills:
            return []
        
        # Usar el top_k configurado si no se especifica
        if top_k is None:
            top_k = self.top_k
        
        # Crear un contexto de consulta enriquecido
        query_context = self._build_query_context(task, agent_state)
        
        # Generar embedding para el contexto de consulta
        query_embedding = self._generate_embedding(query_context)
        
        # Calcular similitudes
        similarities = self._calculate_similarities(query_embedding)
        
        # Obtener los índices de las top_k habilidades más similares
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Recopilar las habilidades relevantes
        relevant_skills = [self.skills[idx] for idx in top_indices]
        
        # Actualizar el contador de usos
        for idx in top_indices:
            self.skills[idx]['uses'] += 1
        
        # Guardar la biblioteca actualizada
        self._save_skills()
        
        return relevant_skills
    
    def _generate_skill_description(self, task: str, actions: List[str], result: str) -> str:
        """
        Genera una descripción concisa de una habilidad.
        
        Args:
            task: Descripción de la tarea
            actions: Lista de acciones/comandos utilizados
            result: Resultado de la ejecución de las acciones
            
        Returns:
            Descripción de la habilidad
        """
        # Construir el prompt
        prompt = f"""
        Genera una descripción concisa de una habilidad en un juego de ficción interactiva.
        
        TAREA:
        {task}
        
        ACCIONES REALIZADAS:
        {', '.join(actions)}
        
        RESULTADO:
        {result}
        
        INSTRUCCIONES:
        1. Crea una descripción clara y concisa de esta habilidad (1-2 oraciones).
        2. La descripción debe ser útil para reutilizar esta habilidad en situaciones similares.
        3. No incluyas detalles específicos como nombres propios o ubicaciones exactas.
        4. Usa un formato general que describa la secuencia de acciones y su propósito.
        5. La descripción debe empezar con un verbo en infinitivo (ej. "Abrir", "Conseguir").
        
        DESCRIPCIÓN DE LA HABILIDAD:
        """
        
        # Generar la descripción
        response = self.llm.generate(prompt, temperature=0.4, max_tokens=100)
        
        # Limpiar la respuesta
        description = response.strip()
        
        # Limitar la longitud de la descripción
        if len(description.split()) > 20:
            words = description.split()
            description = ' '.join(words[:20])
        
        return description
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Genera un embedding vectorial para un texto.
        
        Args:
            text: Texto a embeber
            
        Returns:
            Vector de embedding
        """
        # Usar el modelo de embedding para generar el vector
        embedding = self.embedding_model.encode(text)
        
        # Normalizar el embedding (opcional pero recomendado para comparaciones de similitud)
        normalized_embedding = embedding / np.linalg.norm(embedding)
        
        return normalized_embedding
    
    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calcula similitudes entre un embedding de consulta y los embeddings de la biblioteca.
        
        Args:
            query_embedding: Embedding de la consulta
            
        Returns:
            Array de similitudes
        """
        # Convertir la lista de embeddings a una matriz numpy
        skill_embeddings_matrix = np.array(self.skill_embeddings)
        
        # Calcular la similitud de coseno
        similarities = np.dot(skill_embeddings_matrix, query_embedding)
        
        return similarities
    
    def _build_query_context(self, task: str, agent_state: Dict[str, Any]) -> str:
        """
        Construye un contexto de consulta enriquecido.
        
        Args:
            task: Descripción de la tarea
            agent_state: Estado actual del agente
            
        Returns:
            Contexto de consulta enriquecido
        """
        # Extraer información relevante del estado
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')
        
        # Generar una consulta mejorada que considere el contexto
        prompt = f"""
        Dada una tarea en un juego de ficción interactiva, genera una consulta enriquecida
        que pueda usarse para buscar habilidades relevantes en una biblioteca.
        
        TAREA:
        {task}
        
        OBSERVACIÓN ACTUAL:
        {observation}
        
        INVENTARIO:
        {inventory}
        
        INSTRUCCIONES:
        1. Analiza la tarea y el contexto.
        2. Identifica los objetivos, acciones, objetos y localizaciones clave.
        3. Genera una consulta ampliada que capture la esencia de lo que se necesita hacer.
        4. Incluye sinónimos o términos relacionados para mejorar la recuperación.
        5. Mantén la consulta concisa pero informativa (2-3 oraciones).
        
        CONSULTA ENRIQUECIDA:
        """
        
        # Generar la consulta enriquecida
        response = self.llm.generate(prompt, temperature=0.4, max_tokens=150)
        
        # Limpiar la respuesta
        query_context = response.strip()
        
        return f"{task} {query_context}"
    
    def _save_skills(self) -> None:
        """Guarda la biblioteca de habilidades en el disco."""
        skills_path = os.path.join(self.save_dir, 'skills.json')
        embeddings_path = os.path.join(self.save_dir, 'embeddings.npy')
        
        # Guardar habilidades
        with open(skills_path, 'w') as f:
            json.dump(self.skills, f, indent=2)
        
        # Guardar embeddings
        if self.skill_embeddings:
            np.save(embeddings_path, np.array(self.skill_embeddings))
    
    def _load_skills(self) -> None:
        """Carga la biblioteca de habilidades desde el disco."""
        skills_path = os.path.join(self.save_dir, 'skills.json')
        embeddings_path = os.path.join(self.save_dir, 'embeddings.npy')
        
        # Cargar habilidades si existen
        if os.path.exists(skills_path):
            with open(skills_path, 'r') as f:
                self.skills = json.load(f)
        
        # Cargar embeddings si existen
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            self.skill_embeddings = [embedding for embedding in embeddings]