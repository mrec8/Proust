"""
Main entry point for the narrative agent based on the Voyager architecture.
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

# Configure logging
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
    """Main function that runs the narrative agent."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autonomous Narrative Agent based on Voyager')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file')
    parser.add_argument('--game_config', type=str, default='config/games.yaml', help='Path to the game configuration file')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of steps (overrides config)')
    parser.add_argument('--game', type=str, default=None, help='Game to load (overrides config)')
    parser.add_argument('--autonomous', action='store_true', help='Run in fully autonomous mode')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode (allows user input)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override configuration if arguments are specified
    if args.max_steps:
        config['environment']['max_steps'] = args.max_steps
    if args.game:
        config['environment']['game'] = args.game
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('skills/vector_db', exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Initialize LLM
    llm = LLMInterface(args.config)
    
    # Initialize embedding model
    embedding_model = SentenceTransformer(config['skill_library']['embedding_model'])
    
    # Initialize memory
    memory = Memory()
    
    # Initialize Jericho environment
    env = JerichoEnvironment(config['environment']['game'])
    env.max_steps = config['environment']['max_steps']
    
    # Initialize observation parser
    obs_parser = ObservationParser()
    
    # Initialize agents
    curriculum_agent = CurriculumAgent(args.config, args.game_config, llm)
    action_agent = ActionAgent(args.config, args.game_config, llm)
    critic_agent = CriticAgent(args.config, args.game_config, llm)
    skill_manager = SkillManager(args.config, llm, embedding_model)
    
    # Initialize motivation module
    motivation_module = MotivationModule(args.config, llm, memory)
    
    # Main loop
    logger.info(f"Starting narrative agent in the game {config['environment']['game']}...")
    
    # Initialize environment
    agent_state = env.reset()
    
    # Tracking variables
    steps = 0
    max_steps = config['environment']['max_steps']
    score = 0
    current_task = None
    task_actions = []
    
    # Interactive mode allows the user to input commands or let the agent act
    interactive = args.interactive
    
    try:
        while steps < max_steps:
            # Display current state
            print("\n" + "="*80)
            print(f"Step: {steps}/{max_steps} | Score: {score}")
            print(f"Observation: {agent_state['observation']}")
            print(f"Inventory: {agent_state['inventory']}")
            
            # If there is no current task or the last task was completed/failed
            if current_task is None:
                # Display motivational state
                motivational_state = motivation_module.get_motivational_state()
                print("\nMotivational State:")
                print(f"Role: {motivational_state['role'] or 'Undefined'} (Confidence: {motivational_state['role_confidence']:.2f})")
                print(f"Curiosity: {motivational_state['curiosity']:.2f}, Mastery: {motivational_state['mastery']:.2f}, Autonomy: {motivational_state['autonomy']:.2f}")
                
                # Generate intrinsic goal if in autonomous mode
                if args.autonomous:
                    intrinsic_goal = motivation_module.generate_intrinsic_goal(agent_state)
                    print(f"\nIntrinsic Goal: {intrinsic_goal}")
                    
                # Propose the next task
                current_task = curriculum_agent.propose_next_task(agent_state)
                print(f"\nProposed Task: {current_task}")
                task_actions = []
                
                # In interactive mode, allow the user to accept, reject, or modify the task
                if interactive:
                    user_input = input("\nAccept this task? (y/n/modify): ").strip().lower()
                    if user_input == 'n':
                        current_task = None
                        continue
                    elif user_input.startswith('m'):
                        current_task = input("Enter the modified task: ").strip()
                        print(f"Modified Task: {current_task}")
            
            # Execute a step of the current task
            print(f"\nExecuting task: {current_task}")
            
            # Retrieve relevant skills
            relevant_skills = skill_manager.retrieve_skills(current_task, agent_state)
            if relevant_skills:
                print("\nRelevant skills retrieved:")
                for i, skill in enumerate(relevant_skills):
                    print(f"{i+1}. {skill['description']}")
            
            # Generate action
            action = action_agent.generate_action(current_task, agent_state, relevant_skills)
            print(f"Generated action: {action}")
            
            # In interactive mode, allow the user to accept, reject, or modify the action
            if interactive:
                user_input = input("\nExecute this action? (y/n/modify/manual): ").strip().lower()
                if user_input == 'n':
                    # Refine the action
                    action = action_agent.refine_action(action, current_task, agent_state, 
                                                       "The user rejected this action")
                    print(f"Refined action: {action}")
                elif user_input.startswith('m'):
                    action = input("Enter the modified action: ").strip()
                    print(f"Modified action: {action}")
                elif user_input == 'manual':
                    action = input("Enter a manual command: ").strip()
                    print(f"Manual command: {action}")
            
            # Execute action in the environment
            task_actions.append(action)
            next_state, reward, done, info = env.step(action)
            
            # Update tracking variables
            steps += 1
            old_score = score
            score = next_state['score']
            agent_state = next_state
            
            # Display result
            print(f"\nResult: {agent_state['observation']}")
            if score > old_score:
                print(f"Score increased! +{score - old_score} points")
            
            # Check if the task has been completed
            if steps % 3 == 0 or reward > 0:  # Check every 3 steps or if there is a reward
                success, critique = critic_agent.check_task_success(current_task, agent_state, 
                                                                  [(current_task, a) for a in task_actions])
                
                if success:
                    print(f"\n✅ Task successfully completed: {current_task}")
                    
                    # Add skill to the manager
                    skill = skill_manager.add_skill(current_task, task_actions, 
                                                   agent_state['observation'], True)
                    
                    # Add episode to memory
                    memory.add_episode(current_task, task_actions, agent_state['observation'], 
                                      agent_state, True)
                    
                    # Update curriculum agent
                    curriculum_agent.add_completed_task(current_task)
                    
                    # Update motivation module
                    motivation_module.update_motivation({
                        'task': current_task,
                        'actions': task_actions,
                        'result': agent_state['observation'],
                        'success': True
                    })
                    
                    # Reset task variables
                    current_task = None
                    task_actions = []
                    
                elif len(task_actions) >= 5:  # If too many actions without success
                    print(f"\n❌ Task failed: {current_task}")
                    print(f"Critique: {critique}")
                    
                    # Add episode to memory
                    memory.add_episode(current_task, task_actions, agent_state['observation'], 
                                      agent_state, False)
                    
                    # Update curriculum agent
                    curriculum_agent.add_failed_task(current_task)
                    
                    # Update motivation module
                    motivation_module.update_motivation({
                        'task': current_task,
                        'actions': task_actions,
                        'result': agent_state['observation'],
                        'success': False
                    })
                    
                    # Reset task variables
                    current_task = None
                    task_actions = []
            
            # Check if the game is over
            if done:
                print("\nGame over!")
                break
            
            # Pause for readability
            if not interactive:
                time.sleep(1)
        
        # Display final summary
        print("\n" + "="*80)
        print("Session summary:")
        print(f"Total steps: {steps}")
        print(f"Final score: {score}")
        print(f"Tasks completed: {len(curriculum_agent.get_completed_tasks())}")
        print(f"Tasks failed: {len(curriculum_agent.get_failed_tasks())}")
        
        # Display emerging role
        motivational_state = motivation_module.get_motivational_state()
        if motivational_state['role']:
            print(f"\nEmerging role: {motivational_state['role']} (Confidence: {motivational_state['role_confidence']:.2f})")
            print(motivation_module.get_role_description())
        
        # Display memory summary
        memory_summary = memory.generate_summary()
        print("\nMemory summary:")
        print(f"Episodes: {memory_summary['episodic_memory']['total']} (Successes: {memory_summary['episodic_memory']['successful']}, Failures: {memory_summary['episodic_memory']['failed']})")
        print(f"Learned concepts: {memory_summary['semantic_memory']['concepts']}")
        print(f"Discovered entities: {sum(memory_summary['entity_memory'].values())}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving state...")
    
    finally:
        # Close environment
        env.close()
        print("\nEnvironment closed. Until the next adventure!")

def run_interactive_mode():
    """Runs the agent in interactive mode with greater user control."""
    # Implement in a future version
    pass

if __name__ == "__main__":
    main()