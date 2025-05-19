"""
Script to analyze results from multiple experiments and generate comparison metrics.
"""
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

def load_experiment_metrics(metrics_dir: str) -> List[Dict[str, Any]]:
    """
    Load all experiment metrics from the metrics directory.
    
    Args:
        metrics_dir: Directory containing metrics files
        
    Returns:
        List of loaded metrics (each as a dictionary)
    """
    metrics_files = glob.glob(os.path.join(metrics_dir, '*_metrics.json'))
    loaded_metrics = []
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                loaded_metrics.append(metrics)
                print(f"Loaded metrics from {file_path}")
        except Exception as e:
            print(f"Error loading metrics from {file_path}: {e}")
    
    return loaded_metrics

def extract_summaries(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract summary data from experiments into a DataFrame.
    
    Args:
        experiments: List of experiment metrics
        
    Returns:
        DataFrame with summary metrics
    """
    summaries = []
    
    for exp in experiments:
        # Calculate derived metrics
        duration = (exp["experiment_info"]["end_time"] or 0) - exp["experiment_info"]["start_time"]
        task_success_rate = exp["cumulative_metrics"]["tasks_completed"] / max(1, (
            exp["cumulative_metrics"]["tasks_completed"] + 
            exp["cumulative_metrics"]["tasks_failed"]
        ))
        
        # Extract location and object counts
        #locations_count = len(exp["cumulative_metrics"]["unique_locations_visited"])
        objects_count = len(exp["cumulative_metrics"]["unique_objects_interacted"])
        
        # Create summary dict
        summary = {
            "game": exp["experiment_info"]["game"],
            "experiment_name": exp["experiment_info"]["name"],
            "duration_minutes": duration / 60,
            "total_steps": len(exp["timestep_metrics"]),
            "final_score": exp["cumulative_metrics"]["total_score"],
            "tasks_proposed": exp["cumulative_metrics"]["tasks_proposed"],
            "tasks_completed": exp["cumulative_metrics"]["tasks_completed"],
            "tasks_failed": exp["cumulative_metrics"]["tasks_failed"],
            "skills_acquired": exp["cumulative_metrics"]["skills_acquired"],
            "unique_objects": objects_count,
            "task_success_rate": task_success_rate
        }
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries)

def analyze_task_completion(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze task completion patterns.
    
    Args:
        experiments: List of experiment metrics
        
    Returns:
        DataFrame with task statistics
    """
    all_tasks = []
    
    for exp in experiments:
        game_name = exp["experiment_info"]["game"]
        
        # Process completed tasks
        for task in exp["completed_tasks"]:
            task_info = {
                "game": game_name,
                "task": task["task"],
                "status": "completed",
                "steps_taken": task["steps_taken"],
                "command_count": len(task["commands"]),
            }
            all_tasks.append(task_info)
        
        # Process failed tasks
        for task in exp["failed_tasks"]:
            task_info = {
                "game": game_name,
                "task": task["task"],
                "status": "failed",
                "steps_taken": task["steps_taken"],
                "command_count": len(task["commands"]),
                "reason": task.get("reason", "Unknown")
            }
            all_tasks.append(task_info)
    
    return pd.DataFrame(all_tasks)

def analyze_skill_acquisition(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze skill acquisition patterns.
    
    Args:
        experiments: List of experiment metrics
        
    Returns:
        DataFrame with skill statistics
    """
    all_skills = []
    
    for exp in experiments:
        game_name = exp["experiment_info"]["game"]
        
        for skill in exp["acquired_skills"]:
            skill_info = {
                "game": game_name,
                "skill_description": skill["skill_description"],
                "task": skill["task"],
                #"command_count": len(skill["commands"]),
                "timestamp": skill["timestamp"] - exp["experiment_info"]["start_time"]  # Relative time
            }
            all_skills.append(skill_info)
    
    return pd.DataFrame(all_skills)

def analyze_score_progression(experiments: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """
    Analyze score progression over time.
    
    Args:
        experiments: List of experiment metrics
        
    Returns:
        Dictionary mapping game names to DataFrames with score progression
    """
    game_progressions = {}
    
    for exp in experiments:
        game_name = exp["experiment_info"]["game"]
        
        # Extract timestep data
        timesteps = []
        for step in exp["timestep_metrics"]:
            timestep_info = {
                "step": step["step"],
                "score": step["score"],
                "relative_time": step["timestamp"] - exp["experiment_info"]["start_time"]
            }
            timesteps.append(timestep_info)
        
        # Create DataFrame for this game if it doesn't exist
        if game_name not in game_progressions:
            game_progressions[game_name] = []
        
        # Add experiment data
        game_progressions[game_name].extend(timesteps)
    
    # Convert lists to DataFrames
    for game, data in game_progressions.items():
        game_progressions[game] = pd.DataFrame(data)
    
    return game_progressions

def generate_comparison_plots(summaries_df: pd.DataFrame, 
                             tasks_df: pd.DataFrame,
                             skills_df: pd.DataFrame,
                             progression_dfs: Dict[str, pd.DataFrame],
                             output_dir: str) -> None:
    """
    Generate comparison plots for experiments.
    
    Args:
        summaries_df: DataFrame with experiment summaries
        tasks_df: DataFrame with task statistics
        skills_df: DataFrame with skill statistics
        progression_dfs: Dictionary of DataFrames with score progressions
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11})
    
    # 1. Game performance comparison
    plt.figure(figsize=(10, 6))
    performance_metrics = ['final_score', 'tasks_completed', 'skills_acquired']#, 'unique_locations']
    summary_by_game = summaries_df.groupby('game')[performance_metrics].mean().reset_index()
    
    perf_fig, perf_axes = plt.subplots(2, 2, figsize=(12, 10))
    perf_axes = perf_axes.flatten()
    
    for i, metric in enumerate(performance_metrics):
        sns.barplot(x='game', y=metric, data=summary_by_game, ax=perf_axes[i])
        perf_axes[i].set_title(f'Average {metric.replace("_", " ").title()}')
        perf_axes[i].set_xlabel('Game')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'game_performance_comparison.png'))
    
    # 2. Task completion rates by game
    plt.figure(figsize=(8, 6))
    task_completion = tasks_df.groupby(['game', 'status']).size().unstack().fillna(0)
    task_completion['total'] = task_completion.sum(axis=1)
    task_completion['completion_rate'] = task_completion['completed'] / task_completion['total']
    
    ax = task_completion['completion_rate'].plot(kind='bar')
    ax.set_title('Task Completion Rate by Game')
    ax.set_xlabel('Game')
    ax.set_ylabel('Completion Rate')
    ax.set_ylim(0, 1)
    
    # Add completion rate values on bars
    for i, v in enumerate(task_completion['completion_rate']):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_completion_rates.png'))
    
    # 3. Steps required for task completion
    plt.figure(figsize=(10, 6))
    completed_tasks = tasks_df[tasks_df['status'] == 'completed']
    
    if not completed_tasks.empty:  # Check if there are completed tasks
        sns.boxplot(x='game', y='steps_taken', data=completed_tasks)
        plt.title('Steps Required for Task Completion')
        plt.xlabel('Game')
        plt.ylabel('Steps Taken')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'task_completion_steps.png'))
    
    # 4. Score progression over time
    plt.figure(figsize=(10, 6))
    for game, df in progression_dfs.items():
        if not df.empty:  # Check if there is progression data
            plt.plot(df['step'], df['score'], label=game)
    
    plt.title('Score Progression Over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_progression.png'))
    
    # 5. Skill acquisition rate
    if not skills_df.empty:  # Check if there are skills data
        plt.figure(figsize=(10, 6))
        skills_by_game = skills_df.groupby('game').size()
        ax = skills_by_game.plot(kind='bar')
        ax.set_title('Skills Acquired by Game')
        ax.set_xlabel('Game')
        ax.set_ylabel('Number of Skills')
        
        # Add skill count values on bars
        for i, v in enumerate(skills_by_game):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'skills_by_game.png'))
    
    print(f"Saved comparison plots to {output_dir}")

def generate_detailed_tables(summaries_df: pd.DataFrame, 
                            tasks_df: pd.DataFrame,
                            skills_df: pd.DataFrame,
                            output_dir: str) -> None:
    """
    Generate detailed tables for the analysis.
    
    Args:
        summaries_df: DataFrame with experiment summaries
        tasks_df: DataFrame with task statistics
        skills_df: DataFrame with skill statistics
        output_dir: Directory to save the tables
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Experiment summary table
    summaries_df.to_csv(os.path.join(output_dir, 'experiment_summaries.csv'), index=False)
    
    # 2. Task statistics table
    if not tasks_df.empty:
        tasks_df.to_csv(os.path.join(output_dir, 'task_statistics.csv'), index=False)
        
        # Top 10 completed tasks
        top_completed = tasks_df[tasks_df['status'] == 'completed'].sort_values('steps_taken')
        top_completed.head(10).to_csv(os.path.join(output_dir, 'top10_completed_tasks.csv'), index=False)
        
        # Top 10 failed tasks
        top_failed = tasks_df[tasks_df['status'] == 'failed'].sort_values('steps_taken', ascending=False)
        top_failed.head(10).to_csv(os.path.join(output_dir, 'top10_failed_tasks.csv'), index=False)
    
    # 3. Skill statistics table
    if not skills_df.empty:
        skills_df.to_csv(os.path.join(output_dir, 'skill_statistics.csv'), index=False)
    
    print(f"Saved detailed tables to {output_dir}")

def generate_task_analysis(tasks_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate task analysis reports.
    
    Args:
        tasks_df: DataFrame with task statistics
        output_dir: Directory to save the analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if tasks_df.empty:
        print("No task data to analyze")
        return
    
    # 1. Identify common patterns in task descriptions
    tasks_df['task_length'] = tasks_df['task'].str.len()
    tasks_df['first_word'] = tasks_df['task'].str.split().str[0]
    
    # Analyze verb frequency
    verb_frequency = tasks_df['first_word'].value_counts()
    verb_frequency.to_csv(os.path.join(output_dir, 'verb_frequency.csv'))
    
    # Plot verb frequency
    plt.figure(figsize=(10, 6))
    top_verbs = verb_frequency.head(10)
    ax = top_verbs.plot(kind='bar')
    ax.set_title('Top 10 Task Verbs')
    ax.set_xlabel('Verb')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_verbs.png'))
    
    # 2. Analyze success rates by task type
    task_type_success = tasks_df.groupby(['first_word', 'status']).size().unstack().fillna(0)
    if 'completed' in task_type_success.columns and 'failed' in task_type_success.columns:
        task_type_success['total'] = task_type_success['completed'] + task_type_success['failed']
        task_type_success['success_rate'] = task_type_success['completed'] / task_type_success['total']
        task_type_success = task_type_success.sort_values('success_rate', ascending=False)
        task_type_success.to_csv(os.path.join(output_dir, 'task_type_success.csv'))
        
        # Plot success rates by task type
        plt.figure(figsize=(12, 6))
        ax = task_type_success['success_rate'].head(10).plot(kind='bar')
        ax.set_title('Success Rate by Task Type')
        ax.set_xlabel('Task Type (First Verb)')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'task_type_success.png'))
    
    # 3. Analyze reasons for failure
    if 'reason' in tasks_df.columns:
        failed_tasks = tasks_df[tasks_df['status'] == 'failed']
        
        # Extract key patterns from failure reasons
        failed_tasks['reason_key'] = failed_tasks['reason'].str.extract(r'(cannot|not found|impossible|no path|unclear|ambiguous)', flags=re.IGNORECASE)
        failed_tasks['reason_key'] = failed_tasks['reason_key'].fillna('other')
        
        failure_reasons = failed_tasks['reason_key'].value_counts()
        failure_reasons.to_csv(os.path.join(output_dir, 'failure_reasons.csv'))
        
        # Plot failure reasons
        plt.figure(figsize=(10, 6))
        ax = failure_reasons.plot(kind='pie', autopct='%1.1f%%')
        ax.set_title('Common Failure Reasons')
        ax.set_ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'failure_reasons.png'))
    
    print(f"Saved task analysis to {output_dir}")

def main():
    """Main function to run the analysis."""
    # Configure input/output directories
    metrics_dir = "logs/metrics"
    output_dir = "analysis_results"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment metrics
    print("Loading experiment metrics...")
    experiments = load_experiment_metrics(metrics_dir)
    
    if not experiments:
        print("No experiment metrics found. Please run experiments first.")
        return
    
    # Extract and analyze metrics
    print("Extracting summary metrics...")
    summaries_df = extract_summaries(experiments)
    
    print("Analyzing task completion...")
    tasks_df = analyze_task_completion(experiments)
    
    print("Analyzing skill acquisition...")
    skills_df = analyze_skill_acquisition(experiments)
    
    print("Analyzing score progression...")
    progression_dfs = analyze_score_progression(experiments)
    
    # Generate reports
    print("Generating comparison plots...")
    generate_comparison_plots(
        summaries_df, tasks_df, skills_df, progression_dfs,
        os.path.join(output_dir, "plots")
    )
    
    print("Generating detailed tables...")
    generate_detailed_tables(
        summaries_df, tasks_df, skills_df,
        os.path.join(output_dir, "tables")
    )
    
    print("Generating task analysis...")
    generate_task_analysis(
        tasks_df,
        os.path.join(output_dir, "task_analysis")
    )
    
    print(f"Analysis complete. Results saved to {output_dir}.")

if __name__ == "__main__":
    import re  # For regex in task analysis
    main()