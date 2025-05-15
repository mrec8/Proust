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

# Import our components (to be implemented)
from environment.jericho_wrapper import JerichoEnvironment
from agent.curriculum_agent import CurriculumAgent
from agent.action_agent import ActionAgent
from agent.critic_agent import CriticAgent
from utils.llm_interface import LLMInterface
from agent.memory_manager import MemoryManager
from utils.logging import setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Narrative Agent based on Voyager architecture')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to the configuration file')
    parser.add_argument('--game_config', type=str, default='config/games.yaml', 
                        help='Path to the game configuration file')
    parser.add_argument('--game', type=str, default=None, 
                        help='Game to load (overrides config)')
    parser.add_argument('--max_steps', type=int, default=None, 
                        help='Maximum number of steps (overrides config)')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode (allows user input)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level')
    return parser.parse_args()

def load_configs(args):
    """Load configuration files."""
    # Load main configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load game-specific configuration
    with open(args.game_config, 'r') as f:
        game_config = yaml.safe_load(f)
    
    # Override with command line arguments if specified
    if args.game:
        config['environment']['game'] = args.game
    if args.max_steps:
        config['environment']['max_steps'] = args.max_steps
    
    return config, game_config

def main():
    """Main function that runs the narrative agent."""
    # Parse arguments and load configurations
    args = parse_arguments()
    config, game_config = load_configs(args)
    
    # Set up logging
    log_dir = config.get('logging', {}).get('save_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting narrative agent...")
    
    # Initialize components
    try:
        # LLM Interface
        llm = LLMInterface(config)
        logger.info("LLM interface initialized")
        
        # Environment
        game_name = config['environment']['game']
        env = JerichoEnvironment(game_name)
        env.max_steps = config['environment']['max_steps']
        logger.info(f"Environment initialized with game: {game_name}")
        
        # Skill Manager
        memory_manager = MemoryManager(config, llm, game_name)
        logger.info("Skill Manager initialized")
        
        # Agents
        curriculum_agent = CurriculumAgent(config, game_config, llm)
        action_agent = ActionAgent(config, game_config, llm)
        critic_agent = CriticAgent(config, game_config, llm)
        logger.info("Agents initialized")
        
        # Main loop
        run_agent_loop(env, curriculum_agent, action_agent, critic_agent, 
                      memory_manager, llm, config, args.interactive)
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        return 1
    
    return 0

def run_agent_loop(env, curriculum_agent, action_agent, critic_agent, 
                  memory_manager, llm, config, interactive=False):
    """Run the main agent loop."""
    logger = logging.getLogger(__name__)
    
    # Initialize environment
    agent_state = env.reset()
    
    # Tracking variables
    steps = 0
    max_steps = config['environment']['max_steps']
    current_task = None
    task_actions = []
    

    logger.info("Starting agent loop...")
    
    try:
        while steps < max_steps:
            # Display current state if in interactive mode
            if interactive:
                print("\n" + "="*80)
                print(f"Step: {steps}/{max_steps}")
                print(f"Observation: {agent_state['observation']}")
                print(f"Inventory: {agent_state['inventory']}")
            
            # If there is no current task or the last task was completed/failed
            if current_task is None:

                # Retrieve memories relevant for task planning
                planning_context = f"Current location: {agent_state['observation']}\nInventory: {agent_state['inventory']}"
                relevant_memories = memory_manager.retrieve_memories(planning_context, agent_state)


                # Propose the next task using the curriculum agent with memories
                current_task = curriculum_agent.propose_next_task(agent_state, relevant_memories)
                critique = "None"
                logger.info(f"New task proposed: {current_task}")
                
                if interactive:
                    print(f"\nProposed Task: {current_task}")
                    user_input = input("\nAccept this task? (y/n/modify): ").strip().lower()
                    if user_input == 'n':
                        current_task = None
                        continue
                    elif user_input.startswith('m'):
                        current_task = input("Enter the modified task: ").strip()
                        print(f"Modified Task: {current_task}")
                
                task_actions = []
            
            # Execute a step for the current task
            if interactive:
                print(f"\nExecuting task: {current_task}")
            
            # Retrieve relevant memories
            context = f"Task: {current_task}\nObservation: {agent_state['observation']}\nInventory: {agent_state['inventory']}"
            relevant_memories = memory_manager.retrieve_memories(context, agent_state)
            
            if relevant_memories:
                memory_topics = [memory.topic for memory in relevant_memories]
                logger.info(f"Relevant memories retrieved for task '{current_task}':")
                for i, topic in enumerate(memory_topics):
                    logger.info(f"{i+1}. \"{topic}\"")
                if interactive: 
                    print("\nRelevant memories retrieved:")
                    for i, topic in enumerate(memory_topics):
                        print(f"{i+1}. \"{topic}\"")
            
            # Generate a single action
            action = action_agent.generate_action(current_task, agent_state, critique, relevant_memories)
            logger.info(f"Generated action: '{action}'")
            
            if interactive:
                print(f"Generated action: '{action}'")
                
                # Allow user intervention in interactive mode
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

            if memory_manager.should_create_memory(next_state['observation'], next_state, agent_state['observation']): 
                memory_manager.add_memory(next_state['observation'], next_state)
                logger.info("New memory created")
                if interactive:
                    print("New memory created")
            
            # Log the response from the environment
            #logger.info(f"Environment response: {next_state['observation']}")
            
            # Update tracking variables
            steps += 1
            agent_state = next_state
            
            # Display result if interactive
            if interactive:
                print(f"\nResult: {agent_state['observation']}")
            
            # Check if the task has been completed
            
            success, critique = critic_agent.check_task_success(
                current_task, agent_state, task_actions
            )
            
            if success:
                logger.info(f"Task successfully completed: {current_task}")
                
                # Update curriculum agent
                curriculum_agent.add_completed_task(current_task)
                
                # Reset task variables
                current_task = None
                task_actions = []
                
                if interactive:
                    print(f"\n✅ Task successfully completed: {current_task}")
                    
            elif len(task_actions) >= 5:  # If too many actions without success
                logger.info(f"Task failed: {current_task}")
                logger.info(f"Critique: {critique}")
                
                # Update curriculum agent
                curriculum_agent.add_failed_task(current_task)
                
                # Reset task variables
                current_task = None
                task_actions = []
                
                if interactive:
                    print(f"\n❌ Task failed: {current_task}")
                    print(f"Critique: {critique}")
            
            # Check if the game is over
            if done:
                logger.info("Game over!")
                if interactive:
                    print("\nGame over!")
                break
            
            # Pause for readability in interactive mode
            if interactive:
                time.sleep(1)
        
        # Display final summary
        logger.info(f"Session ended after {steps} steps")
        if interactive:
            print("\n" + "="*80)
            print("Session summary:")
            print(f"Total steps: {steps}")
            print(f"Tasks completed: {len(curriculum_agent.get_completed_tasks())}")
            print(f"Tasks failed: {len(curriculum_agent.get_failed_tasks())}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if interactive:
            print("\n\nInterrupted by user. Saving state...")
    
    finally:
        # Close environment
        env.close()
        logger.info("Environment closed")
        if interactive:
            print("\nEnvironment closed. Until the next adventure!")

if __name__ == "__main__":
    sys.exit(main())