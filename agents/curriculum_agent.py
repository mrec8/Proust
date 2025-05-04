"""
Agente de Currículo Automático para proponer tareas adaptadas al progreso.
"""
import os
import yaml
import json
import random
from typing import Dict, List, Tuple, Set, Any, Optional

from utils.llm_interface import LLMInterface

class CurriculumAgent:
    """
    Agente que propone tareas adaptadas al progreso del agente y al contexto narrativo.
    """
    
    def __init__(self, config_path: str, game_config_path: str, llm: LLMInterface):
        """
        Inicializa el agente de currículo.
        
        Args:
            config_path: Ruta al archivo de configuración
            game_config_path: Ruta al archivo de configuración del juego
            llm: Interfaz con el modelo de lenguaje
        """
        # Cargar configuraciones
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(game_config_path, 'r') as f:
            self.game_config = yaml.safe_load(f)
        
        self.llm = llm
        
        # Configuraciones específicas
        self.game_name = self.config['environment']['game']
        self.game_specific_config = self.game_config.get(self.game_name, {})
        
        # Inicializar histórico de tareas
        self.completed_tasks = []
        self.failed_tasks = []
        self.current_task = None
        
        # Parámetros de dificultad
        self.difficulty_scaling = self.config['agents']['curriculum_agent'].get('task_difficulty_scaling', 1.2)
        self.max_failed_tasks = self.config['agents']['curriculum_agent'].get('max_failed_tasks_memory', 10)
        
        # Inicializar con objetivos del juego si están disponibles
        self.progression_milestones = self.game_specific_config.get('progression_milestones', [])
        
        # Almacenar la tarea inicial si está disponible
        self.initial_goal = self.game_specific_config.get('starting_goal', 
                                                         "Explorar el mundo y descubrir su funcionamiento")
    
    def propose_next_task(self, agent_state: Dict[str, Any]) -> str:
        """
        Propone la siguiente tarea basada en el estado actual y el progreso.
        
        Args:
            agent_state: Estado actual del agente
            
        Returns:
            Descripción de la siguiente tarea a realizar
        """
        # Si es la primera tarea, utilizar el objetivo inicial
        if not self.completed_tasks and not self.failed_tasks:
            self.current_task = self._generate_initial_task()
            return self.current_task
        
        # Obtener información relevante para generar la siguiente tarea
        exploration_progress = self._get_exploration_progress()
        
        # Construir prompt para el LLM
        prompt = self._build_task_proposal_prompt(agent_state, exploration_progress)
        
        # Generar la siguiente tarea usando el LLM
        response = self.llm.generate(prompt, temperature=0.7)
        
        # Extraer la tarea del texto generado
        task = self._extract_task_from_response(response)
        
        # Actualizar la tarea actual
        self.current_task = task
        
        return task
    
    def _generate_initial_task(self) -> str:
        """
        Genera la primera tarea para el agente.
        
        Returns:
            Descripción de la tarea inicial
        """
        # Si hay un objetivo inicial en la configuración, usarlo
        if self.initial_goal:
            # Descomponer el objetivo inicial en una tarea específica
            prompt = f"""
            En el juego '{self.game_name}', el objetivo general es: "{self.initial_goal}".
            
            Como primera tarea específica y concreta para empezar, el agente debería:
            """
            response = self.llm.generate(prompt, temperature=0.5, max_tokens=50)
            initial_task = response.strip()
            
            # Si la respuesta es demasiado genérica, usar una tarea predeterminada
            if len(initial_task.split()) < 3:
                if self.game_name == "zork1":
                    return "Explorar los alrededores de la casa blanca"
                else:
                    return f"Explorar la ubicación inicial y examinar el entorno"
            
            return initial_task
        
        # Si no hay objetivo inicial, generar una tarea inicial básica
        return "Explorar los alrededores y familiarizarse con el entorno"
    
    def _get_exploration_progress(self) -> Dict[str, Any]:
        """
        Calcula métricas de progreso de exploración basadas en tareas completadas y fallidas.
        
        Returns:
            Diccionario con métricas de progreso
        """
        # Contar tareas por categoría
        task_categories = {
            "exploration": 0,  # Exploración de lugares
            "collection": 0,   # Recolección de objetos
            "interaction": 0,  # Interacción con objetos
            "puzzle": 0,       # Resolver puzzles
            "combat": 0,       # Combate con entidades
            "conversation": 0  # Conversación con NPCs
        }
        
        # Clasificar tareas completadas
        for task in self.completed_tasks:
            task_lower = task.lower()
            if any(word in task_lower for word in ["explorar", "ir", "visitar", "encontrar lugar"]):
                task_categories["exploration"] += 1
            elif any(word in task_lower for word in ["obtener", "recoger", "tomar"]):
                task_categories["collection"] += 1
            elif any(word in task_lower for word in ["usar", "abrir", "cerrar", "mover", "empujar", "tirar"]):
                task_categories["interaction"] += 1
            elif any(word in task_lower for word in ["resolver", "desbloquear", "descifrar"]):
                task_categories["puzzle"] += 1
            elif any(word in task_lower for word in ["atacar", "matar", "luchar"]):
                task_categories["combat"] += 1
            elif any(word in task_lower for word in ["hablar", "preguntar", "responder"]):
                task_categories["conversation"] += 1
        
        # Calcular tasa de éxito
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        success_rate = len(self.completed_tasks) / total_tasks if total_tasks > 0 else 0
        
        return {
            "completed_tasks_count": len(self.completed_tasks),
            "failed_tasks_count": len(self.failed_tasks),
            "success_rate": success_rate,
            "task_categories": task_categories,
            "completed_tasks": self.completed_tasks[-5:],  # Últimas 5 tareas completadas
            "failed_tasks": self.failed_tasks[-5:]         # Últimas 5 tareas fallidas
        }
    
    def _build_task_proposal_prompt(self, agent_state: Dict[str, Any], 
                                   exploration_progress: Dict[str, Any]) -> str:
        """
        Construye el prompt para generar la próxima tarea.
        
        Args:
            agent_state: Estado actual del agente
            exploration_progress: Métricas de progreso de exploración
            
        Returns:
            Prompt para el LLM
        """
        # Extraer información relevante del estado
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')
        score = agent_state.get('score', 0)
        moves = agent_state.get('moves', 0)
        
        # Extraer información de progreso
        completed_count = exploration_progress["completed_tasks_count"]
        failed_count = exploration_progress["failed_tasks_count"]
        success_rate = exploration_progress["success_rate"]
        task_categories = exploration_progress["task_categories"]
        recent_completed = exploration_progress["completed_tasks"]
        recent_failed = exploration_progress["failed_tasks"]
        
        # Construir el prompt
        prompt = f"""
        Eres un agente de currículo inteligente para un juego de ficción interactiva llamado {self.game_name}.
        Tu tarea es proponer el siguiente objetivo inmediato para un agente que está explorando este mundo narrativo.
        
        INFORMACIÓN DEL JUEGO:
        {self.game_specific_config.get('description', 'Un juego de aventura textual.')}
        
        ESTADO ACTUAL DEL AGENTE:
        Observación actual: "{observation}"
        Inventario actual: "{inventory}"
        Puntuación actual: {score}
        Movimientos realizados: {moves}
        
        PROGRESO DE EXPLORACIÓN:
        Tareas completadas hasta ahora: {completed_count}
        Tareas fallidas hasta ahora: {failed_count}
        Tasa de éxito: {success_rate:.2f}
        
        Categorías de tareas completadas:
        - Exploración: {task_categories["exploration"]}
        - Recolección: {task_categories["collection"]}
        - Interacción: {task_categories["interaction"]}
        - Puzzles: {task_categories["puzzle"]}
        - Combate: {task_categories["combat"]}
        - Conversación: {task_categories["conversation"]}
        
        Tareas completadas recientemente:
        {', '.join(recent_completed) if recent_completed else 'Ninguna'}
        
        Tareas fallidas recientemente:
        {', '.join(recent_failed) if recent_failed else 'Ninguna'}
        
        CRITERIOS PARA LA PRÓXIMA TAREA:
        1. La tarea debe ser específica, concreta y verificable.
        2. La tarea debe ser alcanzable con el estado y recursos actuales del agente.
        3. La tarea debe adaptarse al nivel de habilidad actual del agente.
        4. La tarea debe contribuir a la exploración y progreso en el juego.
        5. La tarea no debe repetir exactamente algo que el agente acaba de fallar.
        6. La tarea debe mantener coherencia narrativa con el mundo del juego.
        
        INSTRUCCIONES:
        Propón UNA SOLA tarea específica que el agente debería intentar a continuación.
        La tarea debe ser breve y empezar con un verbo en infinitivo (p.ej., "Explorar", "Recoger", "Abrir").
        No incluyas explicaciones, justificaciones ni instrucciones adicionales.
        
        PRÓXIMA TAREA:
        """
        
        return prompt
    
    def _extract_task_from_response(self, response: str) -> str:
        """
        Extrae la tarea del texto generado por el LLM.
        
        Args:
            response: Texto completo generado por el LLM
            
        Returns:
            Tarea extraída
        """
        # Limpiar y formatear la respuesta
        task = response.strip()
        
        # Eliminar prefijos comunes que el LLM podría generar
        prefixes = [
            "La próxima tarea es:",
            "Próxima tarea:",
            "Tarea:",
            "La tarea es:"
        ]
        
        for prefix in prefixes:
            if task.startswith(prefix):
                task = task[len(prefix):].strip()
        
        # Limitar la longitud de la tarea para que sea concisa
        if len(task.split()) > 10:
            # Tomar solo las primeras 10 palabras
            words = task.split()
            task = ' '.join(words[:10])
            
            # Asegurarse de que termina con un signo de puntuación
            if not task.endswith(('.', '!', '?')):
                task += '.'
        
        return task
    
    def add_completed_task(self, task: str) -> None:
        """
        Registra una tarea como completada.
        
        Args:
            task: Descripción de la tarea completada
        """
        # Agregar a la lista de tareas completadas
        self.completed_tasks.append(task)
        
        # Si era la tarea actual, limpiar
        if self.current_task == task:
            self.current_task = None
    
    def add_failed_task(self, task: str) -> None:
        """
        Registra una tarea como fallida.
        
        Args:
            task: Descripción de la tarea fallida
        """
        # Agregar a la lista de tareas fallidas
        self.failed_tasks.append(task)
        
        # Limitar el número de tareas fallidas recordadas
        if len(self.failed_tasks) > self.max_failed_tasks:
            self.failed_tasks = self.failed_tasks[-self.max_failed_tasks:]
        
        # Si era la tarea actual, limpiar
        if self.current_task == task:
            self.current_task = None
    
    def get_completed_tasks(self) -> List[str]:
        """
        Devuelve la lista de tareas completadas.
        
        Returns:
            Lista de tareas completadas
        """
        return self.completed_tasks
    
    def get_failed_tasks(self) -> List[str]:
        """
        Devuelve la lista de tareas fallidas.
        
        Returns:
            Lista de tareas fallidas
        """
        return self.failed_tasks