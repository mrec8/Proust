"""
Módulo para analizar y estructurar las observaciones textuales de Jericho.
"""
import re
from typing import Dict, List, Set, Tuple, Optional, Any

class ObservationParser:
    """
    Clase para parsear y estructurar las observaciones textuales de los juegos Jericho.
    """
    
    def __init__(self):
        """Inicializa el parser con expresiones regulares y patrones comunes."""
        # Patrones para detectar ubicaciones
        self.location_patterns = [
            r"You are in (.*?)\.",
            r"You are standing (.*?)\.",
            r"You're in (.*?)\.",
        ]
        
        # Patrones para detectar objetos
        self.object_patterns = [
            r"You can see (.*?) here\.",
            r"There is (.*?) here\.",
            r"There's (.*?) here\.",
        ]
        
        # Patrones para detectar salidas/direcciones
        self.exit_patterns = [
            r"Exits: (.*?)\.",
            r"You can go: (.*?)\.",
            r"Obvious exits: (.*?)\.",
        ]
        
        # Patrones para detectar resultados de acciones
        self.action_result_patterns = {
            "take": [r"Taken\.", r"You pick up (.*?)\.", r"You take (.*?)\."],
            "drop": [r"Dropped\.", r"You drop (.*?)\.", r"Dropped\."],
            "open": [r"Opened\.", r"You open (.*?)\.", r"The (.*?) is now open\."],
            "close": [r"Closed\.", r"You close (.*?)\.", r"The (.*?) is now closed\."],
            "examine": [r"(.*?) looks (.*?)\.", r"You see nothing special about (.*?)\."],
            "inventory": [r"You are carrying (.*?)\.", r"You're carrying (.*?)\."],
        }
        
        # Lista de objetos comunes en juegos de aventura
        self.common_objects = {
            "sword", "knife", "key", "lamp", "lantern", "book", "scroll", "coin", 
            "gold", "silver", "bottle", "flask", "food", "water", "map", "compass",
            "torch", "door", "window", "chest", "box", "bag", "sack", "rope",
            "letter", "note", "paper", "pen", "pencil", "ring", "necklace", "jewel",
            "gem", "diamond", "ruby", "emerald", "sapphire", "pearl", "crown", "wand",
            "staff", "rod", "cloak", "robe", "armor", "shield", "helmet", "boots",
            "gloves", "gauntlets", "potion", "spell", "magic", "dragon", "monster",
            "creature", "beast", "goblin", "troll", "ogre", "ghost", "skeleton"
        }
        
        # Direcciones comunes
        self.directions = {
            "north", "south", "east", "west", "northeast", "northwest", 
            "southeast", "southwest", "up", "down", "in", "out"
        }
    
    def parse_observation(self, observation: str) -> Dict[str, Any]:
        """
        Analiza una observación textual para extraer información estructurada.
        
        Args:
            observation: Texto de observación del juego
            
        Returns:
            Diccionario con información estructurada extraída de la observación
        """
        result = {
            'raw_text': observation,
            'location': self._extract_location(observation),
            'objects': self._extract_objects(observation),
            'exits': self._extract_exits(observation),
            'action_results': self._extract_action_results(observation),
            'entities': self._extract_entities(observation),
            'messages': self._extract_messages(observation),
        }
        
        return result
    
    def _extract_location(self, text: str) -> str:
        """Extrae la descripción de la ubicación actual."""
        for pattern in self.location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Si no encontramos un patrón explícito, usamos la primera línea como ubicación
        lines = text.split('\n')
        if lines:
            return lines[0].strip()
        
        return ""
    
    def _extract_objects(self, text: str) -> List[str]:
        """Extrae objetos mencionados en la observación."""
        objects = []
        
        # Usando patrones explícitos
        for pattern in self.object_patterns:
            match = re.search(pattern, text)
            if match:
                object_text = match.group(1).strip()
                # Separar objetos si hay listas (separadas por comas y/o "and")
                items = re.split(r',\s*|\s+and\s+', object_text)
                objects.extend(items)
        
        # Buscando objetos comunes en el texto
        for obj in self.common_objects:
            if re.search(r'\b' + obj + r'\b', text.lower()):
                if obj not in objects:
                    objects.append(obj)
        
        return objects
    
    def _extract_exits(self, text: str) -> List[str]:
        """Extrae las salidas o direcciones disponibles."""
        exits = []
        
        # Usando patrones explícitos
        for pattern in self.exit_patterns:
            match = re.search(pattern, text)
            if match:
                exit_text = match.group(1).strip()
                # Separar direcciones si hay listas
                directions = re.split(r',\s*|\s+and\s+|\s+or\s+', exit_text)
                exits.extend(directions)
        
        # Buscando direcciones comunes en el texto
        for direction in self.directions:
            if re.search(r'\b' + direction + r'\b', text.lower()):
                if direction not in exits:
                    exits.append(direction)
        
        return exits
    
    def _extract_action_results(self, text: str) -> Dict[str, str]:
        """Extrae los resultados de acciones específicas."""
        results = {}
        
        for action, patterns in self.action_result_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    if len(match.groups()) > 0:
                        results[action] = match.group(1).strip()
                    else:
                        results[action] = "success"
        
        return results
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extrae entidades o personajes mencionados."""
        # Lista de entidades comunes en juegos de aventura
        entities = [
            "guard", "soldier", "knight", "king", "queen", "prince", "princess",
            "wizard", "witch", "sorcerer", "sorceress", "merchant", "shopkeeper",
            "innkeeper", "bartender", "villager", "farmer", "thief", "rogue",
            "warrior", "fighter", "man", "woman", "child", "boy", "girl",
            "troll", "goblin", "dragon", "elf", "dwarf", "hobbit", "orc", "demon"
        ]
        
        found_entities = []
        for entity in entities:
            if re.search(r'\b' + entity + r'\b', text.lower()):
                found_entities.append(entity)
        
        return found_entities
    
    def _extract_messages(self, text: str) -> List[str]:
        """Extrae mensajes importantes o alertas."""
        # Patrones para mensajes importantes
        message_patterns = [
            r"(\w+) says, \"(.*?)\"",
            r"A voice (.*?) says",
            r"You hear (.*?)\.",
            r"Warning: (.*?)\.",
            r"Suddenly, (.*?)\."
        ]
        
        messages = []
        for pattern in message_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 1:
                    messages.append(f"{match.group(1)}: {match.group(2)}")
                else:
                    messages.append(match.group(1))
        
        return messages
    
    def parse_inventory(self, inventory_text: str) -> List[str]:
        """
        Parsea el texto de inventario para extraer la lista de objetos.
        
        Args:
            inventory_text: Texto que describe el inventario
            
        Returns:
            Lista de objetos en el inventario
        """
        items = []
        
        # Patrones comunes para texto de inventario
        patterns = [
            r"You are carrying (.*?)\.",
            r"You're carrying (.*?)\.",
            r"You have (.*?)\."
        ]
        
        for pattern in patterns:
            match = re.search(pattern, inventory_text)
            if match:
                items_text = match.group(1).strip()
                # Si el texto termina con "nothing", el inventario está vacío
                if items_text.lower() == "nothing":
                    return []
                # Separar objetos si hay listas
                item_list = re.split(r',\s*|\s+and\s+', items_text)
                items.extend(item_list)
                break
        
        # Si no encontramos un patrón explícito, buscamos líneas con objetos
        if not items:
            lines = inventory_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and line.startswith('-') or line.startswith('*'):
                    items.append(line[1:].strip())
        
        return items