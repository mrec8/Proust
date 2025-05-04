"""
Script simplificado para ejecutar el agente narrativo.
"""
import os
import sys
import argparse
import subprocess

def main():
    """Función principal que ejecuta el agente narrativo con opciones simplificadas."""
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Ejecutar el Agente Narrativo Autónomo')
    parser.add_argument('--game', type=str, default='zork1', 
                      help='Juego a ejecutar (por defecto: zork1)')
    parser.add_argument('--steps', type=int, default=100, 
                      help='Número máximo de pasos (por defecto: 100)')
    parser.add_argument('--auto', action='store_true', 
                      help='Ejecutar en modo completamente autónomo')
    parser.add_argument('--interactive', action='store_true',
                      help='Ejecutar en modo interactivo (permite entrada del usuario)')
    args = parser.parse_args()
    
    # Verificar que existen los directorios necesarios
    required_dirs = [
        'config', 'agents', 'environment', 'skills', 'utils', 'logs'
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Error: No se encuentra el directorio '{directory}'")
            print("Asegúrate de estar en el directorio raíz del proyecto.")
            return 1
    
    # Verificar que existen los archivos de configuración
    if not os.path.exists('config/config.yaml'):
        print("Error: No se encuentra el archivo 'config/config.yaml'")
        return 1
    
    if not os.path.exists('config/games.yaml'):
        print("Error: No se encuentra el archivo 'config/games.yaml'")
        return 1
    
    # Construir comando para ejecutar main.py
    cmd = [sys.executable, 'main.py']
    
    # Agregar opciones
    cmd.extend(['--game', args.game])
    cmd.extend(['--max_steps', str(args.steps)])
    
    if args.auto:
        cmd.append('--autonomous')
    
    if args.interactive:
        cmd.append('--interactive')
    
    # Ejecutar el comando
    try:
        subprocess.run(cmd)
        return 0
    except Exception as e:
        print(f"Error al ejecutar el agente: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())