"""
Simplified script to run the narrative agent.
"""
import os
import sys
import argparse
import subprocess

def main():
    """Main function that runs the narrative agent with simplified options."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Autonomous Narrative Agent')
    parser.add_argument('--game', type=str, default='zork1', 
                      help='Game to run (default: zork1)')
    parser.add_argument('--steps', type=int, default=100, 
                      help='Maximum number of steps (default: 100)')
    parser.add_argument('--auto', action='store_true', 
                      help='Run in fully autonomous mode')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode (allows user input)')
    args = parser.parse_args()
    
    # Verify that required directories exist
    required_dirs = [
        'config', 'agents', 'environment', 'skills', 'utils', 'logs'
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' not found")
            print("Make sure you are in the project's root directory.")
            return 1
    
    # Verify that configuration files exist
    if not os.path.exists('config/config.yaml'):
        print("Error: File 'config/config.yaml' not found")
        return 1
    
    if not os.path.exists('config/games.yaml'):
        print("Error: File 'config/games.yaml' not found")
        return 1
    
    # Build command to run main.py
    cmd = [sys.executable, 'main.py']
    
    # Add options
    cmd.extend(['--game', args.game])
    cmd.extend(['--max_steps', str(args.steps)])
    
    if args.auto:
        cmd.append('--autonomous')
    
    if args.interactive:
        cmd.append('--interactive')
    
    # Execute the command
    try:
        subprocess.run(cmd)
        return 0
    except Exception as e:
        print(f"Error running the agent: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())