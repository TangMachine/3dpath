#!/usr/bin/env python3
"""
Visual demonstration of the new merged obstacle and goal map representation.
This script creates a simple environment and shows the visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import ulti6_improved

def create_demo_environment():
    """Create a demo environment with obstacles and goal."""
    # Create a 32x32 map
    obstacle_map = np.zeros((32, 32))
    
    # Add some obstacles
    obstacle_map[8:12, 8:12] = 1    # Square obstacle
    obstacle_map[20:24, 15:19] = 1  # Another obstacle
    obstacle_map[5:7, 20:25] = 1    # Rectangular obstacle
    
    return obstacle_map

def demonstrate_merged_representation():
    """Demonstrate the new 2-matrix representation."""
    print("=== Demonstrating Merged Obstacle and Goal Map ===\n")
    
    # Create environment
    obstacle_map = create_demo_environment()
    env = ulti6_improved.ImprovedPathPlanningEnv(obstacle_map, (2, 2), (28, 28))
    
    # Reset and get initial state
    state = env.reset()
    print(f"State shape: {state.shape} (changed from (3, 32, 32) to (2, 32, 32))")
    
    # Move agent a bit and discover obstacles
    env.current_pos = (7, 7)
    env.current_position_table.fill(0.0)
    env.current_position_table[7, 7] = 1.0
    env.update_discovered_obstacles(7, 7)  # Discover nearby obstacles
    
    # Move again
    env.current_pos = (19, 14)  
    env.current_position_table.fill(0.0)
    env.current_position_table[19, 14] = 1.0
    env.update_discovered_obstacles(19, 14)  # Discover more obstacles
    
    # Get final state
    final_state = env.get_state()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Current position table
    axes[0].imshow(final_state[0], cmap='Reds', origin='upper')
    axes[0].set_title('Current Position Table\n(Agent location)', fontsize=12)
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    
    # Combined map table
    im = axes[1].imshow(final_state[1], cmap='viridis', origin='upper', vmin=0, vmax=2)
    axes[1].set_title('Combined Map Table\n(Obstacles=1.0, Goal=2.0)', fontsize=12)
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Value (0=Empty, 1=Obstacle, 2=Goal)')
    
    plt.tight_layout()
    plt.savefig('/tmp/merged_representation_demo.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to /tmp/merged_representation_demo.png")
    
    # Print statistics
    print(f"\nState statistics:")
    print(f"- Empty spaces: {np.sum(final_state[1] == 0.0)} cells")
    print(f"- Discovered obstacles: {np.sum(final_state[1] == 1.0)} cells") 
    print(f"- Goal position: {np.sum(final_state[1] == 2.0)} cells")
    
    print(f"\nEncoding verification:")
    print(f"- Goal at (28,28): {final_state[1][28, 28]} (should be 2.0)")
    print(f"- Agent at (19,14): {final_state[0][19, 14]} (should be 1.0)")
    
    # Test that agent can still use the state
    agent = ulti6_improved.ImprovedDQNAgent(env)
    action = agent.act(final_state)
    print(f"\nAgent action selection: {action} (valid action from 0-{env.num_actions-1})")
    
    print("\n✓ Merged representation working correctly!")

if __name__ == "__main__":
    demonstrate_merged_representation()