"""
Agente Crítico para la auto-verificación del éxito en tareas.
"""
import os
import yaml
import json
import re
from typing import Dict, List, Tuple, Any, Optional

from utils.llm_interface import LLMInterface
from environment.observation_parser import ObservationParser

class CriticAgent:
    """
    Agente que evalúa el éxito de las tareas y proporciona retroalimentación.
    """
    
    def __init__(self, config_path: str, game_config_path: str, llm: LLMInterface):
        """
        Inicializa el agente crítico.
        
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
        
        # Parámetros del agente crítico
        self.strictness = self.config['agents']['critic_agent'].get('strictness', 0.8)
        
        # Inicializar el parser de observaciones
        self.obs_parser = ObservationParser()
        
        # Historial de evaluaciones
        self.evaluation_history = []
    
    def check_task_success(self, task: str, agent_state: Dict[str, Any], 
                          action_history: Optional[List[Tuple[str, str]]] = None) -> Tuple[bool, str]:
        """
        Verifica si una tarea se ha completado con éxito.
        
        Args:
            task: Tarea a evaluar
            agent_state: Estado actual del agente
            action_history: Historial de acciones (tarea, acción)
            
        Returns:
            Tupla (éxito, crítica)
        """
        # Procesar la observación
        parsed_observation = self.obs_parser.parse_observation(agent_state.get('observation', ''))
        
        # Procesar el inventario
        inventory = agent_state.get('inventory', '')
        parsed_inventory = self.obs_parser.parse_inventory(inventory)
        
        # Extraer información adicional
        
        
        # Construir el prompt para verificación
        prompt = self._build_verification_prompt(task, parsed_observation, 
                                               parsed_inventory, agent_state, 
                                               action_history)
        
        # Generar evaluación
        response = self.llm.generate(prompt, temperature=0.4)
        
        # Extraer resultado de la evaluación
        success, critique = self._extract_evaluation_result(response)
        
        # Registrar la evaluación
        self._add_to_history(task, success, critique)
        
        return success, critique
    
    def _build_verification_prompt(self, task: str, parsed_observation: Dict[str, Any], 
                                  parsed_inventory: List[str], agent_state: Dict[str, Any], 
                                  action_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Construye el prompt para verificar el éxito de una tarea.
        
        Args:
            task: Tarea a evaluar
            parsed_observation: Observación procesada
            parsed_inventory: Inventario procesado
            agent_state: Estado actual del agente
            action_history: Historial de acciones
            
        Returns:
            Prompt para el LLM
        """
        # Extraer información relevante
        location = parsed_observation.get('location', '')
        objects = parsed_observation.get('objects', [])
        exits = parsed_observation.get('exits', [])
        entities = parsed_observation.get('entities', [])
        messages = parsed_observation.get('messages', [])
        
        score = agent_state.get('score', 0)
        moves = agent_state.get('moves', 0)
        
        # Construir historial reciente
        recent_history = ""
        if action_history:
            recent_history = "ACCIONES RECIENTES:\n"
            for i, (t, a) in enumerate(action_history[-5:]):  # Últimas 5 acciones
                recent_history += f"{i+1}. Tarea: {t} -> Acción: {a}\n"
        
        # Construir el prompt
        prompt = f"""
        Eres un crítico experto en juegos de ficción interactiva como {self.game_name}.
        Tu tarea es evaluar si un agente ha completado con éxito una tarea específica.
        Debes ser preciso y estricto en tu evaluación.
        
        TAREA A EVALUAR:
        {task}
        
        ESTADO ACTUAL DEL AGENTE:
        Ubicación: {location}
        Objetos visibles: {', '.join(objects) if objects else 'Ninguno'}
        Salidas: {', '.join(exits) if exits else 'Ninguna visible'}
        Entidades: {', '.join(entities) if entities else 'Ninguna'}
        Mensajes: {', '.join(messages) if messages else 'Ninguno'}
        
        INVENTARIO:
        {', '.join(parsed_inventory) if parsed_inventory else 'Vacío'}
        
        Puntuación: {score}
        Movimientos: {moves}
        
        {recent_history}
        
        INSTRUCCIONES:
        1. Evalúa si la tarea se ha completado con éxito basándote en el estado actual.
        2. Considera los cambios en el inventario, la ubicación, los objetos visibles y los mensajes.
        3. Si la tarea no se ha completado, proporciona una crítica constructiva.
        4. Sé estricto pero justo en tu evaluación.
        
        Responde con el siguiente formato JSON:
        {{
          "razonamiento": "tu análisis detallado",
          "exito": true/false,
          "critica": "crítica constructiva si no tuvo éxito, o vacío si tuvo éxito"
        }}
        """
        
        return prompt
    
    def _extract_evaluation_result(self, response: str) -> Tuple[bool, str]:
        """
        Extrae el resultado de la evaluación del texto generado por el LLM.
        
        Args:
            response: Texto completo generado por el LLM
            
        Returns:
            Tupla (éxito, crítica)
        """
        try:
            # Intentar extraer JSON
            # Buscar el primer '{' y el último '}'
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                result = json.loads(json_str)
                
                success = result.get('exito', False)
                critique = result.get('critica', '')
                
                return success, critique
            
        except json.JSONDecodeError:
            # Si falla la extracción de JSON, intentar un enfoque basado en patrones
            pass
        
        # Enfoque de respaldo basado en patrones
        if "exito: true" in response.lower() or "éxito: true" in response.lower():
            return True, ""
        
        # Buscar una crítica
        critique = ""
        critique_patterns = [
            r"critica: \"(.*?)\"",
            r"crítica: \"(.*?)\"",
            r"crítica:(.*?)(?:\"|$)",
            r"critica:(.*?)(?:\"|$)"
        ]
        
        for pattern in critique_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                critique = match.group(1).strip()
                break
        
        return False, critique
    
    def _add_to_history(self, task: str, success: bool, critique: str) -> None:
        """
        Agrega una evaluación al historial.
        
        Args:
            task: Tarea evaluada
            success: Resultado de la evaluación
            critique: Crítica proporcionada
        """
        self.evaluation_history.append({
            'task': task,
            'success': success,
            'critique': critique
        })
        
        # Limitar el tamaño del historial (mantener las últimas 50 evaluaciones)
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]