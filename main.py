"""
Punto de entrada principal para el agente narrativo basado en la arquitectura Voyager.
"""
import os
import sys
import time
import argparse
import yaml
import logging
from typing import Dict, List, Any, Optional

from sentence_transformers import SentenceTransformer

from environment.jericho_wrapper import JerichoEnvironment
from environment.observation_parser import ObservationParser
from agents.curriculum_agent import CurriculumAgent
from agents.action_agent import ActionAgent
from agents.critic_agent import CriticAgent
from agents.skill_manager import SkillManager
from utils.llm_interface import LLMInterface
from utils.memory import Memory
from utils.motivation import MotivationModule

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Función principal que ejecuta el agente narrativo."""
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Agente Narrativo Autónomo basado en Voyager')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--game_config', type=str, default='config/games.yaml', help='Ruta al archivo de configuración de juegos')
    parser.add_argument('--max_steps', type=int, default=None, help='Número máximo de pasos (sobreescribe config)')
    parser.add_argument('--game', type=str, default=None, help='Juego a cargar (sobreescribe config)')
    parser.add_argument('--autonomous', action='store_true', help='Ejecutar en modo completamente autónomo')
    parser.add_argument('--interactive', action='store_true', help='Ejecutar en modo interactivo (permite entrada del usuario)')
    args = parser.parse_args()
    
    # Cargar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Sobreescribir configuración si se especifican argumentos
    if args.max_steps:
        config['environment']['max_steps'] = args.max_steps
    if args.game:
        config['environment']['game'] = args.game
    
    # Crear directorios necesarios
    os.makedirs('logs', exist_ok=True)
    os.makedirs('skills/vector_db', exist_ok=True)
    
    # Inicializar componentes
    logger.info("Inicializando componentes...")
    
    # Inicializar LLM
    llm = LLMInterface(args.config)
    
    # Inicializar modelo de embedding
    embedding_model = SentenceTransformer(config['skill_library']['embedding_model'])
    
    # Inicializar memoria
    memory = Memory()
    
    # Inicializar entorno Jericho
    env = JerichoEnvironment(config['environment']['game'])
    env.max_steps = config['environment']['max_steps']
    
    # Inicializar parser de observaciones
    obs_parser = ObservationParser()
    
    # Inicializar agentes
    curriculum_agent = CurriculumAgent(args.config, args.game_config, llm)
    action_agent = ActionAgent(args.config, args.game_config, llm)
    critic_agent = CriticAgent(args.config, args.game_config, llm)
    skill_manager = SkillManager(args.config, llm, embedding_model)
    
    # Inicializar módulo de motivación
    motivation_module = MotivationModule(args.config, llm, memory)
    
    # Bucle principal
    logger.info(f"Iniciando agente narrativo en el juego {config['environment']['game']}...")
    
    # Inicializar entorno
    agent_state = env.reset()
    
    # Variables de seguimiento
    steps = 0
    max_steps = config['environment']['max_steps']
    score = 0
    current_task = None
    task_actions = []
    
    # Modo interactivo permite al usuario ingresar comandos o dejar que el agente actúe
    interactive = args.interactive
    
    try:
        while steps < max_steps:
            # Mostrar estado actual
            print("\n" + "="*80)
            print(f"Paso: {steps}/{max_steps} | Puntuación: {score}")
            print(f"Observación: {agent_state['observation']}")
            print(f"Inventario: {agent_state['inventory']}")
            
            # Si no hay tarea actual o la última tarea fue completada/fallida
            if current_task is None:
                # Mostrar estado motivacional
                motivational_state = motivation_module.get_motivational_state()
                print("\nEstado Motivacional:")
                print(f"Rol: {motivational_state['role'] or 'Indefinido'} (Confianza: {motivational_state['role_confidence']:.2f})")
                print(f"Curiosidad: {motivational_state['curiosity']:.2f}, Maestría: {motivational_state['mastery']:.2f}, Autonomía: {motivational_state['autonomy']:.2f}")
                
                # Generar objetivo intrínseco si estamos en modo autónomo
                if args.autonomous:
                    intrinsic_goal = motivation_module.generate_intrinsic_goal(agent_state)
                    print(f"\nObjetivo Intrínseco: {intrinsic_goal}")
                    
                # Proponer la siguiente tarea
                current_task = curriculum_agent.propose_next_task(agent_state)
                print(f"\nTarea Propuesta: {current_task}")
                task_actions = []
                
                # En modo interactivo, permitir al usuario aceptar, rechazar o modificar la tarea
                if interactive:
                    user_input = input("\n¿Aceptar esta tarea? (s/n/modificar): ").strip().lower()
                    if user_input == 'n':
                        current_task = None
                        continue
                    elif user_input.startswith('m'):
                        current_task = input("Ingrese la tarea modificada: ").strip()
                        print(f"Tarea modificada: {current_task}")
            
            # Ejecutar un paso de la tarea actual
            print(f"\nEjecutando tarea: {current_task}")
            
            # Obtener habilidades relevantes
            relevant_skills = skill_manager.retrieve_skills(current_task, agent_state)
            if relevant_skills:
                print("\nHabilidades relevantes recuperadas:")
                for i, skill in enumerate(relevant_skills):
                    print(f"{i+1}. {skill['description']}")
            
            # Generar acción
            action = action_agent.generate_action(current_task, agent_state, relevant_skills)
            print(f"Acción generada: {action}")
            
            # En modo interactivo, permitir al usuario aceptar, rechazar o modificar la acción
            if interactive:
                user_input = input("\n¿Ejecutar esta acción? (s/n/modificar/manual): ").strip().lower()
                if user_input == 'n':
                    # Refinar la acción
                    action = action_agent.refine_action(action, current_task, agent_state, 
                                                       "El usuario rechazó esta acción")
                    print(f"Acción refinada: {action}")
                elif user_input.startswith('m'):
                    action = input("Ingrese la acción modificada: ").strip()
                    print(f"Acción modificada: {action}")
                elif user_input == 'manual':
                    action = input("Ingrese un comando manual: ").strip()
                    print(f"Comando manual: {action}")
            
            # Ejecutar acción en el entorno
            task_actions.append(action)
            next_state, reward, done, info = env.step(action)
            
            # Actualizar variables de seguimiento
            steps += 1
            old_score = score
            score = next_state['score']
            agent_state = next_state
            
            # Mostrar resultado
            print(f"\nResultado: {agent_state['observation']}")
            if score > old_score:
                print(f"¡Puntuación aumentada! +{score - old_score} puntos")
            
            # Verificar si la tarea se ha completado
            if steps % 3 == 0 or reward > 0:  # Verificar cada 3 pasos o si hay recompensa
                success, critique = critic_agent.check_task_success(current_task, agent_state, 
                                                                  [(current_task, a) for a in task_actions])
                
                if success:
                    print(f"\n✅ Tarea completada con éxito: {current_task}")
                    
                    # Agregar habilidad al gestor
                    skill = skill_manager.add_skill(current_task, task_actions, 
                                                   agent_state['observation'], True)
                    
                    # Agregar episodio a la memoria
                    memory.add_episode(current_task, task_actions, agent_state['observation'], 
                                      agent_state, True)
                    
                    # Actualizar agente de currículo
                    curriculum_agent.add_completed_task(current_task)
                    
                    # Actualizar módulo de motivación
                    motivation_module.update_motivation({
                        'task': current_task,
                        'actions': task_actions,
                        'result': agent_state['observation'],
                        'success': True
                    })
                    
                    # Reiniciar variables de tarea
                    current_task = None
                    task_actions = []
                    
                elif len(task_actions) >= 5:  # Si llevamos muchas acciones sin éxito
                    print(f"\n❌ Tarea fallida: {current_task}")
                    print(f"Crítica: {critique}")
                    
                    # Agregar episodio a la memoria
                    memory.add_episode(current_task, task_actions, agent_state['observation'], 
                                      agent_state, False)
                    
                    # Actualizar agente de currículo
                    curriculum_agent.add_failed_task(current_task)
                    
                    # Actualizar módulo de motivación
                    motivation_module.update_motivation({
                        'task': current_task,
                        'actions': task_actions,
                        'result': agent_state['observation'],
                        'success': False
                    })
                    
                    # Reiniciar variables de tarea
                    current_task = None
                    task_actions = []
            
            # Verificar si el juego ha terminado
            if done:
                print("\n¡Juego terminado!")
                break
            
            # Pausa para legibilidad
            if not interactive:
                time.sleep(1)
        
        # Mostrar resumen final
        print("\n" + "="*80)
        print("Resumen de la sesión:")
        print(f"Pasos totales: {steps}")
        print(f"Puntuación final: {score}")
        print(f"Tareas completadas: {len(curriculum_agent.get_completed_tasks())}")
        print(f"Tareas fallidas: {len(curriculum_agent.get_failed_tasks())}")
        
        # Mostrar rol emergente
        motivational_state = motivation_module.get_motivational_state()
        if motivational_state['role']:
            print(f"\nRol emergente: {motivational_state['role']} (Confianza: {motivational_state['role_confidence']:.2f})")
            print(motivation_module.get_role_description())
        
        # Mostrar resumen de memoria
        memory_summary = memory.generate_summary()
        print("\nResumen de memoria:")
        print(f"Episodios: {memory_summary['episodic_memory']['total']} (Éxitos: {memory_summary['episodic_memory']['successful']}, Fallos: {memory_summary['episodic_memory']['failed']})")
        print(f"Conceptos aprendidos: {memory_summary['semantic_memory']['concepts']}")
        print(f"Entidades descubiertas: {sum(memory_summary['entity_memory'].values())}")
        
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario. Guardando estado...")
    
    finally:
        # Cerrar entorno
        env.close()
        print("\nEntorno cerrado. ¡Hasta la próxima aventura!")

def run_interactive_mode():
    """Ejecuta el agente en modo interactivo con mayor control del usuario."""
    # Implementar en una futura versión
    pass

if __name__ == "__main__":
    main()