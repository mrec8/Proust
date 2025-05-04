"""
Sistema de memoria para el agente.
"""
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class Memory:
    """
    Sistema de memoria para almacenar experiencias y conocimientos del agente.
    """
    
    def __init__(self, save_dir: str = 'logs/memory'):
        """
        Inicializa el sistema de memoria.
        
        Args:
            save_dir: Directorio para guardar la memoria
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Inicializar componentes de memoria
        self.episodic_memory = []  # Memoria episódica (experiencias)
        self.semantic_memory = {}  # Memoria semántica (conocimientos)
        self.entity_memory = {}    # Memoria de entidades (objetos, personajes, ubicaciones)
        
        # Cargar memoria existente si hay
        self._load_memory()
    
    def add_episode(self, task: str, actions: List[str], result: str, 
                   agent_state: Dict[str, Any], success: bool) -> None:
        """
        Agrega un episodio a la memoria episódica.
        
        Args:
            task: Tarea realizada
            actions: Lista de acciones tomadas
            result: Resultado de las acciones
            agent_state: Estado del agente al final del episodio
            success: Si la tarea fue exitosa
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
        
        # Extraer conocimientos del episodio
        self._extract_knowledge(episode)
        
        # Guardar memoria
        self._save_memory()
    
    def add_knowledge(self, concept: str, knowledge: str, source: str = 'inference') -> None:
        """
        Agrega conocimiento a la memoria semántica.
        
        Args:
            concept: Concepto o tema del conocimiento
            knowledge: Descripción del conocimiento
            source: Fuente del conocimiento
        """
        # Normalizar el concepto (minúsculas)
        concept_key = concept.lower()
        
        # Crear o actualizar entrada de conocimiento
        if concept_key not in self.semantic_memory:
            self.semantic_memory[concept_key] = []
        
        # Agregar el nuevo conocimiento
        knowledge_entry = {
            'description': knowledge,
            'source': source,
            'timestamp': time.time(),
            'confidence': 0.8  # Valor predeterminado, podría ajustarse
        }
        
        # Evitar duplicados
        for existing in self.semantic_memory[concept_key]:
            if existing['description'] == knowledge:
                # Actualizar confianza y timestamp si ya existe
                existing['confidence'] = max(existing['confidence'], 0.8)
                existing['timestamp'] = time.time()
                return
        
        self.semantic_memory[concept_key].append(knowledge_entry)
        
        # Guardar memoria
        self._save_memory()
    
    def add_entity(self, entity_type: str, name: str, properties: Dict[str, Any]) -> None:
        """
        Agrega una entidad a la memoria de entidades.
        
        Args:
            entity_type: Tipo de entidad (objeto, personaje, ubicación)
            name: Nombre de la entidad
            properties: Propiedades de la entidad
        """
        # Normalizar el nombre (minúsculas)
        entity_key = name.lower()
        
        # Crear estructura si no existe
        if entity_type not in self.entity_memory:
            self.entity_memory[entity_type] = {}
        
        # Crear o actualizar entidad
        if entity_key not in self.entity_memory[entity_type]:
            self.entity_memory[entity_type][entity_key] = {
                'name': name,
                'properties': properties,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'sightings': 1
            }
        else:
            # Actualizar entidad existente
            entity = self.entity_memory[entity_type][entity_key]
            entity['last_seen'] = time.time()
            entity['sightings'] += 1
            
            # Actualizar propiedades (mantener las existentes y agregar nuevas)
            for key, value in properties.items():
                entity['properties'][key] = value
        
        # Guardar memoria
        self._save_memory()
    
    def get_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera recuerdos relevantes para una consulta.
        
        Args:
            query: Texto de consulta
            top_k: Número de recuerdos a recuperar
            
        Returns:
            Lista de recuerdos relevantes
        """
        # Para una implementación simple, buscamos coincidencias de palabras clave
        # En una implementación más avanzada, se usarían embeddings semánticos
        
        # Normalizar la consulta
        query_words = set(query.lower().split())
        
        # Calcular relevancia para cada episodio
        scored_episodes = []
        for episode in self.episodic_memory:
            score = 0
            
            # Buscar en la tarea
            task_words = set(episode['task'].lower().split())
            score += len(query_words.intersection(task_words)) * 2  # Peso mayor para tareas
            
            # Buscar en el resultado
            result_words = set(episode['result'].lower().split())
            score += len(query_words.intersection(result_words))
            
            # Buscar en las acciones
            for action in episode['actions']:
                action_words = set(action.lower().split())
                score += len(query_words.intersection(action_words))
            
            if score > 0:
                scored_episodes.append((episode, score))
        
        # Ordenar por relevancia
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar los top_k más relevantes
        relevant_episodes = [episode for episode, _ in scored_episodes[:top_k]]
        
        return relevant_episodes
    
    def get_knowledge(self, concept: str) -> List[Dict[str, Any]]:
        """
        Recupera conocimientos sobre un concepto.
        
        Args:
            concept: Concepto a buscar
            
        Returns:
            Lista de conocimientos sobre el concepto
        """
        # Normalizar el concepto
        concept_key = concept.lower()
        
        # Buscar conocimientos exactos
        if concept_key in self.semantic_memory:
            return self.semantic_memory[concept_key]
        
        # Buscar conocimientos parciales
        partial_matches = []
        for key, knowledge_list in self.semantic_memory.items():
            # Si el concepto está contenido en la clave o viceversa
            if concept_key in key or key in concept_key:
                # Agregar cada conocimiento con una referencia a su concepto original
                for knowledge in knowledge_list:
                    partial_match = knowledge.copy()
                    partial_match['original_concept'] = key
                    partial_matches.append(partial_match)
        
        return partial_matches
    
    def get_entity(self, entity_type: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Recupera una entidad específica.
        
        Args:
            entity_type: Tipo de entidad
            name: Nombre de la entidad
            
        Returns:
            Información de la entidad o None si no se encuentra
        """
        # Normalizar el nombre
        entity_key = name.lower()
        
        # Verificar si existe
        if entity_type in self.entity_memory and entity_key in self.entity_memory[entity_type]:
            return self.entity_memory[entity_type][entity_key]
        
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Recupera todas las entidades de un tipo específico.
        
        Args:
            entity_type: Tipo de entidad
            
        Returns:
            Lista de entidades del tipo especificado
        """
        if entity_type in self.entity_memory:
            return list(self.entity_memory[entity_type].values())
        
        return []
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Genera un resumen del estado actual de la memoria.
        
        Returns:
            Resumen de la memoria
        """
        # Contar episodios por resultado
        successful_episodes = sum(1 for e in self.episodic_memory if e['success'])
        failed_episodes = len(self.episodic_memory) - successful_episodes
        
        # Contar entidades por tipo
        entity_counts = {entity_type: len(entities) 
                        for entity_type, entities in self.entity_memory.items()}
        
        # Contar conceptos en la memoria semántica
        knowledge_count = len(self.semantic_memory)
        
        # Calcular estadísticas de tiempo
        current_time = time.time()
        if self.episodic_memory:
            first_episode_time = min(e['timestamp'] for e in self.episodic_memory)
            elapsed_time = (current_time - first_episode_time) / 3600  # Horas
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
        Extrae conocimientos e información de entidades de un episodio.
        
        Args:
            episode: Episodio del que extraer conocimientos
        """
        # Ejemplo simple: extraer objetos del inventario como entidades
        if 'inventory' in episode['state']:
            inventory_text = episode['state']['inventory']
            
            # Usar una implementación simple para detectar objetos
            # En una implementación real, se usaría NLP más avanzado
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
        
        # Extraer ubicaciones de la observación
        if 'observation' in episode['state']:
            observation = episode['state']['observation']
            
            # Buscar patrones como "You are in [location]"
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
        """Guarda la memoria en el disco."""
        # Guardar memoria episódica
        episodic_path = os.path.join(self.save_dir, 'episodic_memory.json')
        with open(episodic_path, 'w') as f:
            json.dump(self.episodic_memory, f, indent=2)
        
        # Guardar memoria semántica
        semantic_path = os.path.join(self.save_dir, 'semantic_memory.json')
        with open(semantic_path, 'w') as f:
            json.dump(self.semantic_memory, f, indent=2)
        
        # Guardar memoria de entidades
        entity_path = os.path.join(self.save_dir, 'entity_memory.json')
        with open(entity_path, 'w') as f:
            json.dump(self.entity_memory, f, indent=2)
    
    def _load_memory(self) -> None:
        """Carga la memoria desde el disco."""
        # Cargar memoria episódica
        episodic_path = os.path.join(self.save_dir, 'episodic_memory.json')
        if os.path.exists(episodic_path):
            with open(episodic_path, 'r') as f:
                self.episodic_memory = json.load(f)
        
        # Cargar memoria semántica
        semantic_path = os.path.join(self.save_dir, 'semantic_memory.json')
        if os.path.exists(semantic_path):
            with open(semantic_path, 'r') as f:
                self.semantic_memory = json.load(f)
        
        # Cargar memoria de entidades
        entity_path = os.path.join(self.save_dir, 'entity_memory.json')
        if os.path.exists(entity_path):
            with open(entity_path, 'r') as f:
                self.entity_memory = json.load(f)