"""
Agente de Acción para generar comandos ejecutables en el entorno Jericho.
"""
import os
import re
import yaml
from typing import Dict, List, Tuple, Set, Any, Optional

from utils.llm_interface import LLMInterface
from environment.observation_parser import ObservationParser

class ActionAgent:
    """
    Agente que genera comandos/acciones ejecutables en el entorno Jericho.
    """
    
    def __init__(self, config_path: str, game_config_path: str, llm: LLMInterface):
        """
        Inicializa el agente de acción.
        
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
        
        # Parámetros del agente de acción
        self.max_refinement_iterations = self.config['agents']['action_agent'].get('max_refinement_iterations', 3)
        
        # Inicializar el parser de observaciones
        self.obs_parser = ObservationParser()
        
        # Almacenar comandos especiales del juego
        self.special_commands = self.game_specific_config.get('special_commands', [])
        
        # Historial de acciones
        self.action_history = []
    
    def generate_action(self, task: str, agent_state: Dict[str, Any], 
                       skills: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Genera un comando/acción ejecutable para la tarea dada.
        
        Args:
            task: Tarea actual
            agent_state: Estado actual del agente
            skills: Lista de habilidades relevantes de la biblioteca
            
        Returns:
            Comando/acción ejecutable
        """
        # Procesar la observación
        parsed_observation = self.obs_parser.parse_observation(agent_state.get('observation', ''))
        
        # Procesar el inventario
        inventory = agent_state.get('inventory', '')
        parsed_inventory = self.obs_parser.parse_inventory(inventory)
        
        # Construir prompt para el LLM
        prompt = self._build_action_generation_prompt(task, parsed_observation, 
                                                     parsed_inventory, agent_state, skills)
        
        # Generar acción
        response = self.llm.generate(prompt, temperature=0.7)
        
        # Extraer acción del texto generado
        action = self._extract_action_from_response(response)
        
        # Registrar la acción en el historial
        self._add_to_history(task, action)
        
        return action
    
    def refine_action(self, previous_action: str, task: str, agent_state: Dict[str, Any], 
                     error_message: Optional[str] = None) -> str:
        """
        Refina una acción previa que no tuvo éxito.
        
        Args:
            previous_action: Acción anterior que se desea refinar
            task: Tarea actual
            agent_state: Estado actual del agente
            error_message: Mensaje de error o resultado de la acción anterior
            
        Returns:
            Comando/acción refinado
        """
        # Procesar la observación
        parsed_observation = self.obs_parser.parse_observation(agent_state.get('observation', ''))
        
        # Procesar el inventario
        inventory = agent_state.get('inventory', '')
        parsed_inventory = self.obs_parser.parse_inventory(inventory)
        
        # Construir prompt para refinar
        prompt = self._build_action_refinement_prompt(previous_action, task, 
                                                     parsed_observation, parsed_inventory, 
                                                     agent_state, error_message)
        
        # Generar acción refinada
        response = self.llm.generate(prompt, temperature=0.7)
        
        # Extraer acción del texto generado
        action = self._extract_action_from_response(response)
        
        # Registrar la acción refinada en el historial
        self._add_to_history(task, action, is_refinement=True)
        
        return action
    
    def _build_action_generation_prompt(self, task: str, parsed_observation: Dict[str, Any], 
                                       parsed_inventory: List[str], agent_state: Dict[str, Any], 
                                       skills: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Construye el prompt para generar una acción.
        
        Args:
            task: Tarea actual
            parsed_observation: Observación procesada
            parsed_inventory: Inventario procesado
            agent_state: Estado actual del agente
            skills: Lista de habilidades relevantes
            
        Returns:
            Prompt para el LLM
        """
        # Extraer información relevante
        location = parsed_observation.get('location', '')
        objects = parsed_observation.get('objects', [])
        exits = parsed_observation.get('exits', [])
        entities = parsed_observation.get('entities', [])
        messages = parsed_observation.get('messages', [])
        
        # Construir contexto de habilidades relevantes
        skills_context = ""
        if skills:
            skills_context = "HABILIDADES RELEVANTES:\n"
            for skill in skills:
                skills_context += f"- {skill['description']}\n"
        
        # Construir historial reciente
        recent_history = ""
        if self.action_history:
            recent_history = "ACCIONES RECIENTES:\n"
            for i, (t, a) in enumerate(self.action_history[-5:]):  # Últimas 5 acciones
                recent_history += f"{i+1}. Tarea: {t} -> Acción: {a}\n"
        
        # Construir lista de comandos especiales
        special_commands_str = ""
        if self.special_commands:
            special_commands_str = "COMANDOS ESPECIALES DISPONIBLES:\n"
            special_commands_str += ', '.join(self.special_commands)
        
        # Construir el prompt
        prompt = f"""
        Eres un agente experto en juegos de ficción interactiva como {self.game_name}.
        Tu tarea es generar UN SOLO comando de texto para lograr un objetivo específico.
        
        TAREA ACTUAL:
        {task}
        
        ESTADO ACTUAL:
        Ubicación: {location}
        Objetos visibles: {', '.join(objects) if objects else 'Ninguno'}
        Salidas: {', '.join(exits) if exits else 'Ninguna visible'}
        Entidades: {', '.join(entities) if entities else 'Ninguna'}
        Mensajes: {', '.join(messages) if messages else 'Ninguno'}
        
        INVENTARIO:
        {', '.join(parsed_inventory) if parsed_inventory else 'Vacío'}
        
        {skills_context}
        
        {recent_history}
        
        {special_commands_str}
        
        INSTRUCCIONES:
        1. Genera UN SOLO comando de texto para avanzar hacia el objetivo.
        2. Los comandos deben ser concisos y seguir la sintaxis estándar de juegos de aventura.
        3. Usa verbos en infinitivo seguidos de sustantivos (ej. "examinar llave").
        4. No uses comillas, puntos ni signos de exclamación en el comando.
        5. No incluyas explicaciones ni razonamientos en tu respuesta.
        
        COMANDO:
        """
        
        return prompt
    
    def _build_action_refinement_prompt(self, previous_action: str, task: str, 
                                       parsed_observation: Dict[str, Any], 
                                       parsed_inventory: List[str], 
                                       agent_state: Dict[str, Any], 
                                       error_message: Optional[str] = None) -> str:
        """
        Construye el prompt para refinar una acción previa.
        
        Args:
            previous_action: Acción anterior que se desea refinar
            task: Tarea actual
            parsed_observation: Observación procesada
            parsed_inventory: Inventario procesado
            agent_state: Estado actual del agente
            error_message: Mensaje de error o resultado de la acción anterior
            
        Returns:
            Prompt para el LLM
        """
        # Extraer información relevante
        location = parsed_observation.get('location', '')
        objects = parsed_observation.get('objects', [])
        exits = parsed_observation.get('exits', [])
        entities = parsed_observation.get('entities', [])
        messages = parsed_observation.get('messages', [])
        
        # Construir contexto de error
        error_context = ""
        if error_message:
            error_context = f"""
            RESULTADO DE LA ACCIÓN ANTERIOR:
            Comando: {previous_action}
            Resultado: {error_message}
            """
        
        # Construir el prompt
        prompt = f"""
        Eres un agente experto en juegos de ficción interactiva como {self.game_name}.
        Tu tarea es REFINAR un comando previo que no tuvo éxito.
        
        TAREA ACTUAL:
        {task}
        
        COMANDO ANTERIOR:
        {previous_action}
        
        {error_context}
        
        ESTADO ACTUAL:
        Ubicación: {location}
        Objetos visibles: {', '.join(objects) if objects else 'Ninguno'}
        Salidas: {', '.join(exits) if exits else 'Ninguna visible'}
        Entidades: {', '.join(entities) if entities else 'Ninguna'}
        Mensajes: {', '.join(messages) if messages else 'Ninguno'}
        
        INVENTARIO:
        {', '.join(parsed_inventory) if parsed_inventory else 'Vacío'}
        
        INSTRUCCIONES PARA REFINAMIENTO:
        1. Analiza por qué el comando anterior no funcionó.
        2. Genera UN SOLO comando alternativo que pueda funcionar mejor.
        3. Los comandos deben ser concisos y seguir la sintaxis estándar de juegos de aventura.
        4. Prueba diferentes verbos o nombres si es necesario.
        5. No uses comillas, puntos ni signos de exclamación en el comando.
        6. No incluyas explicaciones ni razonamientos en tu respuesta.
        
        COMANDO REFINADO:
        """
        
        return prompt
    
    def _extract_action_from_response(self, response: str) -> str:
        """
        Extrae la acción del texto generado por el LLM.
        
        Args:
            response: Texto completo generado por el LLM
            
        Returns:
            Acción extraída
        """
        # Limpiar y formatear la respuesta
        action = response.strip()
        
        # Eliminar prefijos comunes que el LLM podría generar
        prefixes = [
            "El comando es:",
            "Comando:",
            "Acción:",
            "Comando refinado:",
            ">",
            "$"
        ]
        
        for prefix in prefixes:
            if action.startswith(prefix):
                action = action[len(prefix):].strip()
        
        # Eliminar comillas si las hay
        action = action.strip('"\'')
        
        # Eliminar puntuación final
        action = re.sub(r'[.!?]$', '', action).strip()
        
        # Convertir a minúsculas para consistencia
        action = action.lower()
        
        return action
    
    def _add_to_history(self, task: str, action: str, is_refinement: bool = False) -> None:
        """
        Agrega una acción al historial.
        
        Args:
            task: Tarea para la que se generó la acción
            action: Acción generada
            is_refinement: Indica si es un refinamiento de una acción previa
        """
        self.action_history.append((task, action))
        
        # Limitar el tamaño del historial (mantener las últimas 100 acciones)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]