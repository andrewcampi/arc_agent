#!/usr/bin/env python3
"""
Script to find the 10 smallest ARC tasks based on total number of grid elements.
"""

import json
from typing import Dict, List, Tuple


def count_grid_elements(grid: List[List[int]]) -> int:
    """Count the total number of elements in a 2D grid."""
    return sum(len(row) for row in grid)


def count_task_elements(task_data: Dict) -> int:
    """Count total elements across all training examples in a task."""
    total_elements = 0
    
    # Count elements in training examples
    for example in task_data.get('train', []):
        # Count input grid elements
        if 'input' in example:
            total_elements += count_grid_elements(example['input'])
        
        # Count output grid elements
        if 'output' in example:
            total_elements += count_grid_elements(example['output'])
    
    return total_elements


def find_smallest_tasks(challenges_file: str, num_tasks: int = 10) -> List[Tuple[str, int]]:
    """
    Find the smallest ARC tasks by total number of grid elements.
    
    Args:
        challenges_file: Path to the ARC challenges JSON file
        num_tasks: Number of smallest tasks to return (default: 10)
    
    Returns:
        List of tuples containing (task_id, total_elements) for the smallest tasks
    """
    print(f"Loading challenges from {challenges_file}...")
    
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    
    print(f"Loaded {len(challenges)} tasks")
    
    # Calculate element counts for each task
    task_sizes = []
    for task_id, task_data in challenges.items():
        element_count = count_task_elements(task_data)
        task_sizes.append((task_id, element_count))
    
    # Sort by element count (ascending) and take the smallest
    task_sizes.sort(key=lambda x: x[1])
    smallest_tasks = task_sizes[:num_tasks]
    
    return smallest_tasks


def main():
    """Main function to find and display the 10 smallest ARC tasks."""
    challenges_file = "public_dataset/arc-agi_training_challenges.json"
    
    try:
        smallest_tasks = find_smallest_tasks(challenges_file, 30)
        
        print("\n" + "="*60)
        print("10 SMALLEST ARC TASKS BY TOTAL GRID ELEMENTS")
        print("="*60)
        
        for i, (task_id, element_count) in enumerate(smallest_tasks, 1):
            print(f"{i:2d}. Task ID: {task_id} | Total Elements: {element_count}")
        
        print("\n" + "="*60)
        print("TASK IDs ONLY:")
        print("="*60)
        
        task_ids = [task_id for task_id, _ in smallest_tasks]
        print(task_ids)
        
    except FileNotFoundError:
        print(f"Error: Could not find {challenges_file}")
        print("Make sure you're running this script from the project root directory.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {challenges_file}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()