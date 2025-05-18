# Proust: A Narrative Exploration Agent

Proust is an intelligent agent designed for narrative exploration in text-based adventure games. Built on a framework inspired by the Voyager architecture, Proust focuses on autonomous exploration, skill acquisition, and goal-oriented behavior in interactive fiction environments like Zork.

## Overview

Proust combines several advanced components to create an agent capable of exploring and interacting with text-based worlds:

- **Automatic Curriculum**: Dynamically proposes appropriate tasks based on the agent's current capabilities and environment state
- **Skill Library**: Accumulates and retrieves skills to solve increasingly complex problems
- **Observation Parser**: Analyzes and structures textual observations from the game
- **Memory System**: Stores episodic experiences, semantic knowledge, and entity information
- **Motivation Module**: Provides intrinsic motivation for exploration and goal-setting

## Features

- Built on Jericho, a framework for interacting with Z-machine games
- Uses Large Language Models (LLMs) for decision-making
- Retains memory of past experiences and learned skills
- Self-verifies task completion
- Supports interactive mode for user intervention
- Autonomous exploration without human guidance

## Installation

### Prerequisites

- Python 3.8+
- Jericho compatible text adventure games

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/proust.git
   cd proust
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```
   Alternatively, create a `.env` file with the following content:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Place Jericho game files in the `games/roms` directory (or use the default games that come with Jericho)

### Game Installation

The repository comes with configuration for two games:
- `zork1`: A classic adventure game where you explore an underground world
- For other games in the current repository: See the `config/games.yaml` file

To add a new game:

1. Obtain the game file (typically a .z5, .z6, or .z8 file)
2. Place it in the `games/roms` directory:
   ```
   mkdir -p games/roms
   cd games/roms
   wget http://example.com/path/to/game.z5
   ```
3. Update the `config/games.yaml` file with a new entry for your game, including:
   - Description
   - Starting goal
   - Key locations, objects, and characters
   - Special commands
   - Progression milestones

## Configuration

The project uses YAML configuration files:

- `config/config.yaml`: General configuration for the agent
- `config/games.yaml`: Game-specific configuration

Modify these files to adjust the agent's behavior, preferred LLM model, and game-specific settings.

## Usage

### Basic Usage

Run the agent with default settings:

```
python main.py
```

This will use the game specified in `config/config.yaml`.

### Command Line Options

```
python main.py [OPTIONS]
```

Available options:

- `--config PATH`: Path to the configuration file (default: `config/config.yaml`)
- `--game_config PATH`: Path to the game configuration file (default: `config/games.yaml`)
- `--game GAME_NAME`: Specify the game to load (overrides config)
- `--max_steps N`: Maximum number of steps (overrides config)
- `--interactive`: Run in interactive mode (allows user input)
- `--log_level LEVEL`: Set logging level (default: INFO)

### Interactive Mode

The interactive mode allows you to:
- Review and modify proposed tasks
- Accept, reject, or modify generated actions
- Enter manual commands
- See real-time agent observations and decisions

Run in interactive mode:

```
python main.py --interactive
```

### Run with Specific Game

Run the agent with a specific game:

```
python main.py --game zork1
```

### Simplified Runner

For a more straightforward interface, use the run_agent.py script:

```
python run_agent.py --game zork1 --steps 200 --interactive
```

## Architecture

Proust consists of several key components:

1. **Environment**: Interfaces with Jericho to interact with text adventure games
2. **Curriculum Agent**: Proposes appropriate tasks based on game state and agent progress
3. **Action Agent**: Generates executable commands to accomplish tasks
4. **Critic Agent**: Verifies task completion and provides feedback
5. **Skill Manager**: Stores and retrieves skills for reuse
6. **Memory System**: Maintains episodic, semantic, and entity memory
7. **Motivation Module**: Provides intrinsic goals and emerging role identity

## Advanced Configuration

### Adding New Games

The game configurations in `config/games.yaml` provide crucial information for the agent to interact effectively with each game:

```yaml
game_name:
  description: "Brief description of the game"
  starting_goal: "Initial objective for the agent"
  initial_inventory: ["item1", "item2"]
  key_locations: ["Location1", "Location2"]
  key_objects: ["object1", "object2"]
  key_characters: ["character1", "character2"]
  special_commands: ["command1", "command2"]
  progression_milestones:
    - name: "First milestone"
      score: 10
    - name: "Second milestone"
      score: 25
```

When adding a new game:

1. Create an entry with the appropriate game name
2. Fill in as many details as possible to help the agent understand the game
3. For the best experience, include special commands specific to the game
4. Consider adding progression milestones to help the curriculum agent assess progress

The quality of game-specific configuration significantly impacts the agent's performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the Voyager architecture for autonomous agents
- Built on Jericho, a framework for text-based games
- Utilizes OpenAI's GPT models for natural language understanding and generation
