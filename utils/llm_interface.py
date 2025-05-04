"""
Interfaz para interactuar con modelos de lenguaje grande (LLM).
"""
import os
import time
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import openai

class LLMInterface:
    """
    Clase para interactuar con modelos de lenguaje grande.
    """
    
    def __init__(self, config_path: str):
        """
        Inicializa la interfaz LLM.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        # Cargar variables de entorno
        load_dotenv()
        
        # Obtener la clave API
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No se encontró la clave API de OpenAI. Establece la variable de entorno OPENAI_API_KEY.")
        
        # Configurar cliente
        openai.api_key = self.api_key
        
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Configuraciones específicas
        self.model = self.config['agents'].get('llm_model', 'gpt-3.5-turbo')
        self.default_temperature = self.config['agents'].get('temperature', 0.7)
        self.default_max_tokens = self.config['agents'].get('max_tokens', 1024)
        
        # Inicializar contador de llamadas y caché
        self.call_count = 0
        self.cache = {}
        
        # Configurar parámetros de tasa de llamadas
        self.rate_limit_per_minute = 20  # Ajustar según tu plan
        self.last_call_timestamp = 0
    
    def generate(self, prompt: str, temperature: Optional[float] = None, 
                max_tokens: Optional[int] = None, use_cache: bool = True) -> str:
        """
        Genera texto usando el LLM.
        
        Args:
            prompt: Texto de entrada para el modelo
            temperature: Temperatura para la generación (mayor valor = más aleatorio)
            max_tokens: Número máximo de tokens a generar
            use_cache: Si se debe usar caché para prompts idénticos
            
        Returns:
            Texto generado por el modelo
        """
        # Usar valores predeterminados si no se especifican
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Verificar si el resultado está en caché
        cache_key = f"{prompt}_{temp}_{tokens}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Gestión de tasas de llamadas
        self._manage_rate_limit()
        
        try:
            # Realizar la llamada a la API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un asistente útil y preciso."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=tokens
            )
            
            # Extraer el texto generado
            generated_text = response.choices[0].message.content
            
            # Actualizar contador y timestamp
            self.call_count += 1
            self.last_call_timestamp = time.time()
            
            # Guardar en caché
            if use_cache:
                self.cache[cache_key] = generated_text
            
            return generated_text
        
        except Exception as e:
            print(f"Error al generar texto: {e}")
            # Retornar un mensaje de error o una respuesta predeterminada
            return f"Error: No se pudo generar texto debido a: {str(e)}"
    
    def _manage_rate_limit(self) -> None:
        """
        Gestiona la tasa de llamadas a la API para evitar exceder límites.
        """
        # Calcular tiempo transcurrido desde la última llamada
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_timestamp
        
        # Si han pasado menos de 60 segundos y ya se han realizado varias llamadas
        if time_since_last_call < 60 and self.call_count >= self.rate_limit_per_minute:
            # Calcular tiempo de espera necesario
            wait_time = 60 - time_since_last_call
            print(f"Esperando {wait_time:.2f} segundos para respetar límites de API...")
            time.sleep(wait_time)
        
        # Reiniciar contador si ha pasado más de un minuto
        if time_since_last_call >= 60:
            self.call_count = 0