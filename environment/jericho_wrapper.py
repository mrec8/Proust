"""
Módulo para interactuar con el entorno Jericho.
"""
import os
import gym
import numpy as np
import jericho
from typing import Dict, List, Tuple, Optional, Any

class JerichoEnvironment:
    """
    Wrapper para interactuar con juegos de ficción interactiva a través de Jericho.
    """
    
    def __init__(self, game_name: str, seed: int = 0):
        """
        Inicializa el entorno Jericho con el juego especificado.
        
        Args:
            game_name: Nombre del juego a cargar
            seed: Semilla para reproducibilidad
        """
        # Construir la ruta al archivo del juego
        self.game_path = os.path.join(jericho.DATA_PATH, f"{game_name}.z5")
        if not os.path.exists(self.game_path):
            self.game_path = os.path.join(jericho.DATA_PATH, f"{game_name}.z6")
            if not os.path.exists(self.game_path):
                raise FileNotFoundError(f"No se encontró el juego: {game_name}")
        
        # Inicializar el entorno de Jericho
        self.env = gym.make(f"jericho/{game_name}-v0")
        self.env.seed(seed)
        
        # Almacenar información sobre el juego actual
        self.game_name = game_name
        self.max_word_length = self.env.get_dictionary_max_length()
        self.vocab = self.env.get_dictionary()
        
        # Historial y estado
        self.steps = 0
        self.max_steps = 100  # Valor por defecto, puede ser actualizado
        self.history = []
        self.initial_observation = None
        self.current_observation = None
        self.current_score = 0
        self.done = False
        
    def reset(self) -> Dict[str, Any]:
        """
        Reinicia el entorno y devuelve la observación inicial.
        
        Returns:
            Estado inicial con observación, inventario, puntuación, etc.
        """
        obs, info = self.env.reset()
        self.steps = 0
        self.history = []
        self.current_score = info['score']
        self.done = False
        
        # Obtenemos información más completa
        valid_actions = self.env.get_valid_actions()
        inventory = self._get_inventory()
        
        # Guardamos la observación inicial
        self.initial_observation = obs
        self.current_observation = obs
        
        # Creamos un estado enriquecido
        state = {
            'observation': obs,
            'inventory': inventory,
            'score': self.current_score,
            'moves': self.steps,
            'valid_actions': valid_actions,
            'game_over': self.done
        }
        
        self.history.append({
            'action': 'RESET',
            'observation': obs,
            'score': self.current_score
        })
        
        return state
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Ejecuta una acción en el entorno y devuelve el nuevo estado.
        
        Args:
            action: Comando de texto a ejecutar
            
        Returns:
            Tupla (state, reward, done, info) con el nuevo estado, recompensa, 
            si ha terminado y información adicional
        """
        # Ejecutamos la acción
        observation, score, done, info = self.env.step(action)
        
        # Incrementamos el contador de pasos
        self.steps += 1
        
        # Calculamos la recompensa (diferencia de puntuación)
        reward = score - self.current_score
        self.current_score = score
        self.done = done
        
        # Actualizamos la observación actual
        self.current_observation = observation
        
        # Obtenemos más información
        valid_actions = self.env.get_valid_actions()
        inventory = self._get_inventory()
        
        # Creamos el estado enriquecido
        state = {
            'observation': observation,
            'inventory': inventory,
            'score': score,
            'moves': self.steps,
            'valid_actions': valid_actions,
            'game_over': done
        }
        
        # Guardamos la acción y observación en el historial
        self.history.append({
            'action': action,
            'observation': observation,
            'score': score
        })
        
        # Si alcanzamos el máximo de pasos, terminamos
        if self.steps >= self.max_steps:
            done = True
            info['reason'] = 'max_steps'
        
        return state, reward, done, info
    
    def _get_inventory(self) -> str:
        """
        Obtiene el inventario actual del jugador.
        
        Returns:
            Texto que describe el inventario
        """
        # Guardamos observación actual
        current_obs = self.current_observation
        
        # Ejecutamos el comando de inventario
        obs, _, _, _ = self.env.step("inventory")
        
        # Restauramos el estado (Jericho no afecta al estado con 'inventory')
        self.current_observation = current_obs
        
        return obs
    
    def get_valid_actions(self) -> List[str]:
        """
        Devuelve una lista de acciones válidas en el estado actual.
        
        Returns:
            Lista de comandos válidos
        """
        return self.env.get_valid_actions()
    
    def get_world_state_description(self) -> Dict[str, Any]:
        """
        Genera una descripción rica del estado actual del mundo.
        
        Returns:
            Diccionario con información detallada del estado
        """
        # Ejecutamos varios comandos para obtener más información sobre el estado
        current_obs = self.current_observation
        
        # Comando "look" para ver alrededor
        look_obs, _, _, _ = self.env.step("look")
        
        # Comando "inventory" para ver inventario
        inv_obs, _, _, _ = self.env.step("inventory")
        
        # Restauramos el estado
        self.current_observation = current_obs
        
        return {
            'room_description': look_obs,
            'inventory': inv_obs,
            'score': self.current_score,
            'moves': self.steps,
            'history': self.history[-5:] if len(self.history) > 5 else self.history  # últimas 5 acciones
        }
    
    def close(self):
        """Cierra el entorno."""
        self.env.close()