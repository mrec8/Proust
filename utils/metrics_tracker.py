"""
Metrics collection and tracking for the narrative agent.
"""
import os
import time
import json
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

class MetricsTracker:
    """
    Class to track and store metrics for the narrative agent.
    """
    
    def __init__(self, save_dir: str = 'logs/metrics', experiment_name: Optional[str] = None, model_name = "gpt-4.1-nano"):
        """
        Initialize the metrics tracker.
        
        Args:
            save_dir: Directory to save the metrics
            experiment_name: Name of the experiment (defaults to timestamp)
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up save directory
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set experiment name
        self.experiment_name = experiment_name or f"run_{int(time.time())}"
        
        # Initialize metrics
        self.metrics = {
            "experiment_info": {
                "name": self.experiment_name,
                "start_time": time.time(),
                "end_time": None,
                "game": None,
                "max_steps": None,
                "model": model_name
            },
            "cumulative_metrics": {
                "tasks_proposed": 0,
                "tasks_completed": 0,
                "tasks_failed": 0,
                "memories_acquired": 0,
                "commands_executed": 0,
                "successful_commands": 0,
                "total_score": 0,
                "unique_objects_interacted": set()
            },
            "timestep_metrics": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "acquired_memories": []
        }
        
        self.logger.info(f"Metrics tracker initialized for experiment: {self.experiment_name}")
    
    def set_experiment_info(self, game: str, max_steps: int) -> None:
        """
        Set basic experiment information.
        
        Args:
            game: Name of the game being played
            max_steps: Maximum number of steps for the experiment
        """
        self.metrics["experiment_info"]["game"] = game
        self.metrics["experiment_info"]["max_steps"] = max_steps
        
    def log_timestep(self, step: int, action: str, observation: str, 
                   score: float, task: Optional[str] = None) -> None:
        """
        Log metrics for a single timestep.
        
        Args:
            step: Current step number
            action: Action executed
            observation: Observation from the environment
            score: Current score
            task: Current task (if any)
        """
        # Update cumulative metrics
        self.metrics["cumulative_metrics"]["commands_executed"] += 1
        self.metrics["cumulative_metrics"]["total_score"] = score
        
        
        # Extract objects if possible (simplified)
        objects = self._extract_objects(observation)
        for obj in objects:
            if obj not in self.metrics["cumulative_metrics"]["unique_objects_interacted"]:
                self.metrics["cumulative_metrics"]["unique_objects_interacted"].add(obj)
        
        # Log timestep
        self.metrics["timestep_metrics"].append({
            "step": step,
            "action": action,
            "task": task,
            "score": score,
            "objects_in_scope": objects,
            "timestamp": time.time()
        })
        
    def log_task_proposed(self, task: str) -> None:
        """
        Log when a new task is proposed.
        
        Args:
            task: The proposed task
        """
        self.metrics["cumulative_metrics"]["tasks_proposed"] += 1
        
    def log_task_completed(self, task: str, steps_taken: int, commands: List[str]) -> None:
        """
        Log when a task is completed.
        
        Args:
            task: The completed task
            steps_taken: Number of steps taken to complete the task
            commands: List of commands used to complete the task
        """
        self.metrics["cumulative_metrics"]["tasks_completed"] += 1
        self.metrics["completed_tasks"].append({
            "task": task,
            "steps_taken": steps_taken,
            "commands": commands,
            "timestamp": time.time()
        })
        
    def log_task_failed(self, task: str, steps_taken: int, commands: List[str], reason: str) -> None:
        """
        Log when a task fails.
        
        Args:
            task: The failed task
            steps_taken: Number of steps taken before failure
            commands: List of commands used
            reason: Reason for failure
        """
        self.metrics["cumulative_metrics"]["tasks_failed"] += 1
        self.metrics["failed_tasks"].append({
            "task": task,
            "steps_taken": steps_taken,
            "commands": commands,
            "reason": reason,
            "timestamp": time.time()
        })
        
    def log_memory_created(self, memory_name: str, task: str) -> None:
        """
        Log when a new memory is acquired.
        
        Args:
            memory_name: Name or description of the memory
            task: Task that led to acquiring the memory
        """
        self.metrics["cumulative_metrics"]["memories_acquired"] += 1
        self.metrics["acquired_memories"].append({
            "memory_name": memory_name,
            "task": task,
            "timestamp": time.time()
        })
        
    def end_experiment(self) -> None:
        """
        Mark the experiment as ended and save final metrics.
        """
        self.metrics["experiment_info"]["end_time"] = time.time()
        self.save_metrics()
        
    def save_metrics(self) -> None:
        """
        Save metrics to a file.
        """
        # Convert set to list for JSON serialization
        metrics_copy = self.metrics.copy()
        
        metrics_copy["cumulative_metrics"]["unique_objects_interacted"] = list(
            self.metrics["cumulative_metrics"]["unique_objects_interacted"]
        )
        
        # Create filename
        filename = f"{self.experiment_name}_metrics.json"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics_copy, f, indent=2)
            self.logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        duration = (self.metrics["experiment_info"]["end_time"] or time.time()) - \
                  self.metrics["experiment_info"]["start_time"]
        
        return {
            "experiment_name": self.metrics["experiment_info"]["name"],
            "game": self.metrics["experiment_info"]["game"],
            "duration_minutes": duration / 60,
            "total_steps": len(self.metrics["timestep_metrics"]),
            "final_score": self.metrics["cumulative_metrics"]["total_score"],
            "tasks_completed": self.metrics["cumulative_metrics"]["tasks_completed"],
            "tasks_failed": self.metrics["cumulative_metrics"]["tasks_failed"],
            "memories_acquired": self.metrics["cumulative_metrics"]["memories_acquired"],
            "unique_objects": len(self.metrics["cumulative_metrics"]["unique_objects_interacted"]),
            "task_success_rate": self.metrics["cumulative_metrics"]["tasks_completed"] / 
                              max(1, (self.metrics["cumulative_metrics"]["tasks_completed"] + 
                                   self.metrics["cumulative_metrics"]["tasks_failed"]))
        }
        
    def print_summary(self) -> None:
        """
        Print a summary of the metrics.
        """
        summary = self.generate_summary()
        
        print("\n" + "="*50)
        print(f"EXPERIMENT SUMMARY: {summary['experiment_name']}")
        print("="*50)
        print(f"Game: {summary['game']}")
        print(f"Duration: {summary['duration_minutes']:.2f} minutes")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Final Score: {summary['final_score']}")
        print("\nTask Performance:")
        print(f"  Tasks Completed: {summary['tasks_completed']}")
        print(f"  Tasks Failed: {summary['tasks_failed']}")
        print(f"  Success Rate: {summary['task_success_rate']*100:.1f}%")
        print("\nExploration:")
        print(f"  Unique Objects: {summary['unique_objects']}")
        print(f"  memories Acquired: {summary['memories_acquired']}")
        print("="*50)
        
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Generate plots for the metrics.
        
        Args:
            save_path: Path to save the plots (optional)
        """
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Score over time
        steps = [entry["step"] for entry in self.metrics["timestep_metrics"]]
        scores = [entry["score"] for entry in self.metrics["timestep_metrics"]]
        axs[0, 0].plot(steps, scores)
        axs[0, 0].set_title('Score Progression')
        axs[0, 0].set_xlabel('Steps')
        axs[0, 0].set_ylabel('Score')
        axs[0, 0].grid(True)
        
        # Plot 2: Tasks completed/failed
        task_counts = {
            'Completed': self.metrics["cumulative_metrics"]["tasks_completed"],
            'Failed': self.metrics["cumulative_metrics"]["tasks_failed"]
        }
        axs[0, 1].bar(task_counts.keys(), task_counts.values())
        axs[0, 1].set_title('Task Performance')
        axs[0, 1].set_ylabel('Count')
        
        # Plot 3: Cumulative memories acquired
        memory_times = [entry["timestamp"] for entry in self.metrics["acquired_memories"]]
        memory_indices = list(range(1, len(memory_times) + 1))
        
        if memory_indices:  # Only plot if there are memories
            axs[1, 0].step(memory_times, memory_indices)
            axs[1, 0].set_title('Memories Acquisition Over Time')
            axs[1, 0].set_xlabel('Time (s)')
            axs[1, 0].set_ylabel('Cumulative memories')
            
        # Plot 4: Object exploration
        exploration = {
            'Objects': len(self.metrics["cumulative_metrics"]["unique_objects_interacted"])
        }
        axs[1, 1].bar(exploration.keys(), exploration.values())
        axs[1, 1].set_title('Exploration Metrics')
        axs[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Metrics plot saved to {save_path}")
            
        plt.show()
        
    def _extract_location(self, observation: str) -> Optional[str]:
        """
        Simple heuristic to extract location from observation.
        This is a simplified approach - for actual implementation,
        use the observation parser.
        
        Args:
            observation: Text observation from the environment
            
        Returns:
            Extracted location or None
        """
        # Simple heuristic: Use the first line as location
        lines = observation.strip().split('\n')
        if lines:
            return lines[0].strip()
        return None
    
    def _extract_objects(self, observation: str) -> List[str]:
        """
        Simple heuristic to extract objects from observation.
        This is a simplified approach - for actual implementation,
        use the observation parser.
        
        Args:
            observation: Text observation from the environment
            
        Returns:
            List of extracted objects
        """
        # Simple heuristic: Look for common object words
        common_objects = [
            "sword", "knife", "key", "lamp", "lantern", "book", "scroll", "coin", 
            "gold", "silver", "bottle", "flask", "food", "water", "map", "compass",
            "torch", "door", "window", "chest", "box", "bag", "sack", "rope",
            "letter", "note", "paper", "pen", "pencil", "ring", "necklace", "jewel"
        ]
        
        found_objects = []
        observation_lower = observation.lower()
        
        for obj in common_objects:
            if obj in observation_lower:
                found_objects.append(obj)
                
        return found_objects