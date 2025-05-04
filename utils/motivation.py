"""
Módulo de motivación para proporcionar autonomía al agente.
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
    Módulo que gestiona la motivación intrínseca y la autonomía del agente.
    """
    
    def __init__(self, config_path: str, llm: LLMInterface, memory: Memory):
        """
        Inicializa el módulo de motivación.
        
        Args:
            config_path: Ruta al archivo de configuración
            llm: Interfaz con el modelo de lenguaje
            memory: Sistema de memoria del agente
        """
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm = llm
        self.memory = memory
        
        # Inicializar componentes de motivación
        self.curiosity = 0.8  # Nivel inicial de curiosidad (0-1)
        self.mastery = 0.2    # Nivel inicial de maestría (0-1)
        self.autonomy = 0.5   # Nivel inicial de autonomía (0-1)
        
        # Rol emergente
        self.role = None
        self.role_confidence = 0.0
        self.potential_roles = []
        
        # Sistema de valores
        self.values = {
            'exploration': 0.8,     # Preferencia por explorar vs. explotar
            'risk_taking': 0.5,     # Preferencia por tomar riesgos
            'social': 0.6,          # Preferencia por interacción social
            'achievement': 0.7,     # Preferencia por logros y progreso
            'collection': 0.5       # Preferencia por coleccionar objetos
        }
        
        # Cargar estado previo si existe
        self._load_state()
    
    def update_motivation(self, task_result: Dict[str, Any]) -> None:
        """
        Actualiza los niveles de motivación basado en resultados de tareas.
        
        Args:
            task_result: Información sobre el resultado de una tarea
        """
        task = task_result.get('task', '')
        success = task_result.get('success', False)
        result = task_result.get('result', '')
        
        # Actualizar curiosidad (disminuye con éxitos repetidos, aumenta con fracasos)
        if success:
            # Buscar tareas similares en memoria
            similar_tasks = self._find_similar_tasks(task)
            
            # Si ya hemos tenido éxito en tareas similares, la curiosidad disminuye
            if similar_tasks:
                self.curiosity = max(0.2, self.curiosity - 0.05)
            else:
                # Nuevo tipo de éxito, aumenta ligeramente la curiosidad
                self.curiosity = min(1.0, self.curiosity + 0.02)
        else:
            # Los fracasos aumentan la curiosidad (queremos entender por qué fallamos)
            self.curiosity = min(1.0, self.curiosity + 0.05)
        
        # Actualizar maestría (aumenta con éxitos, disminuye con fracasos)
        if success:
            self.mastery = min(1.0, self.mastery + 0.03)
        else:
            self.mastery = max(0.1, self.mastery - 0.02)
        
        # Actualizar autonomía (varía según los resultados)
        if success:
            self.autonomy = min(1.0, self.autonomy + 0.02)
        else:
            # Si fallamos pero es en algo nuevo, mantenemos la autonomía
            if 'nunca antes visto' in result.lower() or 'desconocido' in result.lower():
                pass  # Mantener el nivel actual
            else:
                self.autonomy = max(0.3, self.autonomy - 0.01)
        
        # Actualizar sistema de valores basado en la tarea
        self._update_values(task, success, result)
        
        # Actualizar rol emergente
        self._update_role(task, success, result)
        
        # Guardar estado
        self._save_state()
    
    def generate_intrinsic_goal(self, agent_state: Dict[str, Any]) -> str:
        """
        Genera un objetivo intrínseco basado en la motivación actual.
        
        Args:
            agent_state: Estado actual del agente
            
        Returns:
            Objetivo intrínseco generado
        """
        # Extraer información relevante
        observation = agent_state.get('observation', '')
        inventory = agent_state.get('inventory', '')
        score = agent_state.get('score', 0)
        
        # Construir prompt para generar objetivo
        role_context = f"Tu rol emergente es: {self.role}" if self.role else "Aún no tienes un rol definido, estás en fase de exploración."
        
        prompt = f"""
        Eres un agente autónomo en un juego de ficción interactiva. Genera un objetivo intrínseco basado en tu motivación actual y estado.
        
        TU ESTADO MOTIVACIONAL:
        Curiosidad: {self.curiosity:.2f} (Mayor valor = más interés en lo desconocido)
        Maestría: {self.mastery:.2f} (Mayor valor = más confianza en tus habilidades)
        Autonomía: {self.autonomy:.2f} (Mayor valor = mayor iniciativa propia)
        
        TU ROL EMERGENTE:
        {role_context}
        
        TUS VALORES:
        Exploración: {self.values['exploration']:.2f}
        Toma de riesgos: {self.values['risk_taking']:.2f}
        Interacción social: {self.values['social']:.2f}
        Logros: {self.values['achievement']:.2f}
        Coleccionismo: {self.values['collection']:.2f}
        
        TU SITUACIÓN ACTUAL:
        Observación: "{observation}"
        Inventario: "{inventory}"
        Puntuación: {score}
        
        INSTRUCCIONES:
        1. Genera UN SOLO objetivo intrínseco que refleje tu motivación actual y sea coherente con tu rol emergente.
        2. El objetivo debe ser específico, alcanzable y motivado internamente (no por una recompensa externa).
        3. El objetivo debe ser alineado con tus valores personales.
        4. Utiliza un formato de primera persona, como "Quiero explorar..." o "Deseo encontrar...".
        
        OBJETIVO INTRÍNSECO:
        """
        
        # Generar objetivo
        response = self.llm.generate(prompt, temperature=0.7, max_tokens=100)
        
        # Limpiar y formatear la respuesta
        goal = response.strip()
        
        # Limitar la longitud
        if len(goal.split()) > 20:
            goal = ' '.join(goal.split()[:20])
        
        return goal
    
    def get_motivational_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado motivacional actual.
        
        Returns:
            Diccionario con el estado motivacional
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
        Genera una descripción del rol emergente.
        
        Returns:
            Descripción del rol emergente
        """
        if not self.role or self.role_confidence < 0.3:
            return "Todavía estás explorando y descubriendo tu lugar en este mundo. No has adoptado un rol específico."
        
        # Construir prompt para descripción del rol
        prompt = f"""
        Describe el rol "{self.role}" para un personaje en un juego de ficción interactiva.
        
        El personaje tiene los siguientes valores:
        Exploración: {self.values['exploration']:.2f}
        Toma de riesgos: {self.values['risk_taking']:.2f}
        Interacción social: {self.values['social']:.2f}
        Logros: {self.values['achievement']:.2f}
        Coleccionismo: {self.values['collection']:.2f}
        
        INSTRUCCIONES:
        1. Describe este rol en 2-3 oraciones, usando primera persona.
        2. Incluye motivaciones principales y habilidades características.
        3. Menciona cómo este rol ve e interactúa con el mundo.
        
        DESCRIPCIÓN DEL ROL:
        """
        
        # Generar descripción
        response = self.llm.generate(prompt, temperature=0.6, max_tokens=150)
        
        return response.strip()
    
    def _find_similar_tasks(self, task: str) -> List[Dict[str, Any]]:
        """
        Busca tareas similares en la memoria episódica.
        
        Args:
            task: Tarea a comparar
            
        Returns:
            Lista de tareas similares
        """
        return self.memory.get_relevant_memories(task, top_k=3)
    
    def _update_values(self, task: str, success: bool, result: str) -> None:
        """
        Actualiza el sistema de valores basado en la tarea completada.
        
        Args:
            task: Tarea realizada
            success: Si la tarea fue exitosa
            result: Resultado de la tarea
        """
        # Actualizar valor de exploración
        if 'explorar' in task.lower() or 'descubrir' in task.lower():
            if success:
                self.values['exploration'] = min(1.0, self.values['exploration'] + 0.02)
        
        # Actualizar valor de toma de riesgos
        if 'peligroso' in task.lower() or 'riesgo' in task.lower() or 'atacar' in task.lower():
            if success:
                self.values['risk_taking'] = min(1.0, self.values['risk_taking'] + 0.02)
            else:
                self.values['risk_taking'] = max(0.1, self.values['risk_taking'] - 0.01)
        
        # Actualizar valor social
        if 'hablar' in task.lower() or 'preguntar' in task.lower() or 'conversación' in task.lower():
            if success:
                self.values['social'] = min(1.0, self.values['social'] + 0.02)
        
        # Actualizar valor de logros
        if success:
            self.values['achievement'] = min(1.0, self.values['achievement'] + 0.01)
        
        # Actualizar valor de coleccionismo
        if 'recoger' in task.lower() or 'obtener' in task.lower() or 'tomar' in task.lower():
            if success:
                self.values['collection'] = min(1.0, self.values['collection'] + 0.02)
    
    def _update_role(self, task: str, success: bool, result: str) -> None:
        """
        Actualiza el rol emergente basado en la tarea completada.
        
        Args:
            task: Tarea realizada
            success: Si la tarea fue exitosa
            result: Resultado de la tarea
        """
        # Si aún no tenemos un rol definido o la confianza es baja, intentar inferir uno
        if not self.role or self.role_confidence < 0.5:
            # Construir contexto para inferencia de rol
            episodes = self.memory.episodic_memory[-20:]  # Últimos 20 episodios
            
            episodes_summary = ""
            for i, episode in enumerate(episodes):
                episodes_summary += f"{i+1}. Tarea: {episode['task']}, Éxito: {episode['success']}\n"
            
            prompt = f"""
            Analiza el historial de tareas de un personaje en un juego de ficción interactiva para inferir su rol emergente.
            
            HISTORIAL DE TAREAS:
            {episodes_summary}
            
            VALORES DEL PERSONAJE:
            Exploración: {self.values['exploration']:.2f}
            Toma de riesgos: {self.values['risk_taking']:.2f}
            Interacción social: {self.values['social']:.2f}
            Logros: {self.values['achievement']:.2f}
            Coleccionismo: {self.values['collection']:.2f}
            
            INSTRUCCIONES:
            1. Basándote en el historial y valores, infiere el rol más probable para este personaje.
            2. El rol debe ser un sustantivo o frase nominal corta (ej. "Explorador", "Cazador de tesoros", "Diplomático").
            3. Proporciona una breve justificación.
            4. Asigna una confianza entre 0.0 y 1.0 a tu inferencia.
            
            RESPUESTA EN FORMATO JSON:
            {
                "rol": "nombre del rol",
                "justificacion": "breve justificación",
                "confianza": valor_numérico
            }
            """
            
            # Generar inferencia de rol
            response = self.llm.generate(prompt, temperature=0.4, max_tokens=200)
            
            try:
                # Extraer JSON
                import re
                json_match = re.search(r'{.*}', response, re.DOTALL)
                if json_match:
                    role_info = json.loads(json_match.group(0))
                    
                    inferred_role = role_info.get('rol', '')
                    confidence = float(role_info.get('confianza', 0.3))
                    
                    # Actualizar roles potenciales
                    if inferred_role and inferred_role not in [r['role'] for r in self.potential_roles]:
                        self.potential_roles.append({
                            'role': inferred_role,
                            'confidence': confidence,
                            'timestamp': time.time()
                        })
                    
                    # Si la confianza es suficiente, actualizar el rol
                    if confidence > self.role_confidence:
                        self.role = inferred_role
                        self.role_confidence = confidence
            except:
                pass  # Si hay un error, mantener el rol actual
        else:
            # Si ya tenemos un rol establecido, reforzarlo si la tarea es coherente
            role_lower = self.role.lower()
            task_lower = task.lower()
            
            # Verificar si la tarea refuerza el rol actual
            reinforces_role = False
            
            # Ejemplos de refuerzos de rol
            if 'explorador' in role_lower or 'aventurero' in role_lower:
                if 'explorar' in task_lower or 'descubrir' in task_lower:
                    reinforces_role = True
            
            elif 'guerrero' in role_lower or 'cazador' in role_lower:
                if 'atacar' in task_lower or 'luchar' in task_lower or 'matar' in task_lower:
                    reinforces_role = True
            
            elif 'erudito' in role_lower or 'investigador' in role_lower:
                if 'leer' in task_lower or 'estudiar' in task_lower or 'examinar' in task_lower:
                    reinforces_role = True
            
            # Si la tarea refuerza el rol y es exitosa, aumentar la confianza
            if reinforces_role and success:
                self.role_confidence = min(1.0, self.role_confidence + 0.05)
            # Si la tarea contradice el rol, disminuir ligeramente la confianza
            elif not reinforces_role:
                self.role_confidence = max(0.3, self.role_confidence - 0.01)
    
    def _save_state(self) -> None:
        """Guarda el estado motivacional en el disco."""
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
        """Carga el estado motivacional desde el disco."""
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