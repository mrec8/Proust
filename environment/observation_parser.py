"""
Module to analyze and structure textual observations from Jericho.
"""
import re
from typing import Dict, List, Set, Tuple, Optional, Any

class ObservationParser:
    """
    Class to parse and structure textual observations from Jericho games.
    """
    
    def __init__(self):
        """Initializes the parser with regular expressions and common patterns."""
        # Patterns to detect locations
        self.location_patterns = [
            r"You are in (.*?)\.",
            r"You are standing (.*?)\.",
            r"You're in (.*?)\.",
        ]
        
        # Patterns to detect objects
        self.object_patterns = [
            r"You can see (.*?) here\.",
            r"There is (.*?) here\.",
            r"There's (.*?) here\.",
        ]
        
        # Patterns to detect exits/directions
        self.exit_patterns = [
            r"Exits: (.*?)\.",
            r"You can go: (.*?)\.",
            r"Obvious exits: (.*?)\.",
        ]
        
        # Patterns to detect action results
        self.action_result_patterns = {
            "take": [r"Taken\.", r"You pick up (.*?)\.", r"You take (.*?)\."],
            "drop": [r"Dropped\.", r"You drop (.*?)\.", r"Dropped\."],
            "open": [r"Opened\.", r"You open (.*?)\.", r"The (.*?) is now open\."],
            "close": [r"Closed\.", r"You close (.*?)\.", r"The (.*?) is now closed\."],
            "examine": [r"(.*?) looks (.*?)\.", r"You see nothing special about (.*?)\."],
            "inventory": [r"You are carrying (.*?)\.", r"You're carrying (.*?)\."],
        }
        
        # List of common objects in adventure games
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
        
        # Common directions
        self.directions = {
            "north", "south", "east", "west", "northeast", "northwest", 
            "southeast", "southwest", "up", "down", "in", "out"
        }
    
    def parse_observation(self, observation: str) -> Dict[str, Any]:
        """
        Analyzes a textual observation to extract structured information.
        
        Args:
            observation: Text observation from the game
            
        Returns:
            Dictionary with structured information extracted from the observation
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
        """Extracts the description of the current location."""
        for pattern in self.location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # If no explicit pattern is found, use the first line as the location
        lines = text.split('\n')
        if lines:
            return lines[0].strip()
        
        return ""
    
    def _extract_objects(self, text: str) -> List[str]:
        """Extracts objects mentioned in the observation."""
        objects = []
        
        # Using explicit patterns
        for pattern in self.object_patterns:
            match = re.search(pattern, text)
            if match:
                object_text = match.group(1).strip()
                # Separate objects if there are lists (separated by commas and/or "and")
                items = re.split(r',\s*|\s+and\s+', object_text)
                objects.extend(items)
        
        # Searching for common objects in the text
        for obj in self.common_objects:
            if re.search(r'\b' + obj + r'\b', text.lower()):
                if obj not in objects:
                    objects.append(obj)
        
        return objects
    
    def _extract_exits(self, text: str) -> List[str]:
        """Extracts available exits or directions."""
        exits = []
        
        # Using explicit patterns
        for pattern in self.exit_patterns:
            match = re.search(pattern, text)
            if match:
                exit_text = match.group(1).strip()
                # Separate directions if there are lists
                directions = re.split(r',\s*|\s+and\s+|\s+or\s+', exit_text)
                exits.extend(directions)
        
        # Searching for common directions in the text
        for direction in self.directions:
            if re.search(r'\b' + direction + r'\b', text.lower()):
                if direction not in exits:
                    exits.append(direction)
        
        return exits
    
    def _extract_action_results(self, text: str) -> Dict[str, str]:
        """Extracts the results of specific actions."""
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
        """Extracts entities or characters mentioned."""
        # List of common entities in adventure games
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
        """Extracts important messages or alerts."""
        # Patterns for important messages
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
        Parses the inventory text to extract the list of objects.
        
        Args:
            inventory_text: Text describing the inventory
            
        Returns:
            List of objects in the inventory
        """
        items = []
        
        # Common patterns for inventory text
        patterns = [
            r"You are carrying (.*?)\.",
            r"You're carrying (.*?)\.",
            r"You have (.*?)\."
        ]
        
        for pattern in patterns:
            match = re.search(pattern, inventory_text)
            if match:
                items_text = match.group(1).strip()
                # If the text ends with "nothing", the inventory is empty
                if items_text.lower() == "nothing":
                    return []
                # Separate objects if there are lists
                item_list = re.split(r',\s*|\s+and\s+', items_text)
                items.extend(item_list)
                break
        
        # If no explicit pattern is found, look for lines with objects
        if not items:
            lines = inventory_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and line.startswith('-') or line.startswith('*'):
                    items.append(line[1:].strip())
        
        return items