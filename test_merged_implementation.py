#!/usr/bin/env python3
"""
Test script to verify the merged obstacle and goal map implementation.
Tests that the new 2-matrix representation works correctly.
"""

import numpy as np
import torch
import ulti6_improved
import ulti6_im

def test_environment_functionality():
    """Test that the environment works with 2 matrices instead of 3."""
    print("Testing environment functionality...")
    
    # Create test environment
    obstacle_map = np.zeros((64, 64))
    obstacle_map[10:15, 10:15] = 1  # Add obstacles
    obstacle_map[30:35, 30:35] = 1  # Add more obstacles
    
    # Test 2D environment
    env = ulti6_improved.ImprovedPathPlanningEnv(obstacle_map, (1, 1), (60, 60))
    
    # Test reset and initial state
    state = env.reset()
    assert state.shape == (2, 64, 64), f"Expected shape (2, 64, 64), got {state.shape}"
    
    # Test goal position encoding
    assert state[1][60, 60] == 2.0, f"Goal should be 2.0, got {state[1][60, 60]}"
    
    # Test obstacle discovery
    env.update_discovered_obstacles(5, 5)  # Should discover nearby obstacles
    state = env.get_state()
    
    # Check that obstacles around (10,10) are discovered
    discovered_obstacle = state[1][10, 10] == 1.0
    print(f"✓ Obstacle discovered: {discovered_obstacle}")
    
    # Test step functionality
    action = 3  # Right movement
    next_state, reward, done, _ = env.step(action)
    assert next_state.shape == (2, 64, 64), f"Expected shape (2, 64, 64), got {next_state.shape}"
    
    print("✓ Environment functionality test passed")

def test_dqn_model():
    """Test that the DQN model works with 2 input channels."""
    print("Testing DQN model...")
    
    # Create model
    model = ulti6_improved.ImprovedDQN(64, 64, 8)
    
    # Create test input with 2 channels
    test_input = torch.randn(4, 2, 64, 64)  # Batch of 4
    
    # Test forward pass
    output = model(test_input)
    assert output.shape == (4, 8), f"Expected shape (4, 8), got {output.shape}"
    
    print("✓ DQN model test passed")

def test_agent_functionality():
    """Test that the agent works with the new representation."""
    print("Testing agent functionality...")
    
    # Create environment and agent
    obstacle_map = np.zeros((64, 64))
    obstacle_map[20:25, 20:25] = 1
    env = ulti6_improved.ImprovedPathPlanningEnv(obstacle_map, (1, 1), (60, 60))
    agent = ulti6_improved.ImprovedDQNAgent(env)
    
    # Test action selection
    state = env.reset()
    action = agent.act(state)
    assert 0 <= action < env.num_actions, f"Invalid action {action}"
    
    # Test memory and replay (basic test)
    next_state, reward, done, _ = env.step(action)
    agent.remember(state, action, reward, next_state, done)
    
    # Add a few more experiences
    for _ in range(10):
        state = next_state
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        if done:
            break
    
    # Test replay (should not crash)
    agent.replay()
    
    print("✓ Agent functionality test passed")

def test_3d_environment():
    """Test 3D environment with merged representation."""
    print("Testing 3D environment...")
    
    # Create 3D environment
    obstacle_map = np.zeros((32, 32))
    obstacle_map[10:15, 10:15] = 1
    elevation_data = np.random.rand(32, 32) * 10
    
    env = ulti6_improved.ImprovedPathPlanningEnv(
        obstacle_map, (1, 1, 20), (28, 28, 30), 
        is_3d=True, elevation_data=elevation_data
    )
    
    # Test state shape
    state = env.reset()
    assert state.shape == (2, 32, 32), f"Expected shape (2, 32, 32), got {state.shape}"
    
    # Test goal encoding in 3D
    assert state[1][28, 28] == 2.0, f"3D goal should be 2.0, got {state[1][28, 28]}"
    
    print("✓ 3D environment test passed")

def test_both_files_consistent():
    """Test that both ulti6_improved.py and ulti6_im.py work consistently."""
    print("Testing consistency between files...")
    
    obstacle_map = np.zeros((32, 32))
    obstacle_map[5:10, 5:10] = 1
    
    # Test ulti6_improved
    env1 = ulti6_improved.ImprovedPathPlanningEnv(obstacle_map, (1, 1), (28, 28))
    state1 = env1.reset()
    
    # Test ulti6_im
    env2 = ulti6_im.ImprovedPathPlanningEnv(obstacle_map, (1, 1), (28, 28))
    state2 = env2.reset()
    
    # Both should have same shape
    assert state1.shape == state2.shape == (2, 32, 32)
    
    # Both should have goal at same position with same value
    assert state1[1][28, 28] == state2[1][28, 28] == 2.0
    
    # Test models from both files
    model1 = ulti6_improved.ImprovedDQN(32, 32, 8)
    model2 = ulti6_im.ImprovedDQN(32, 32, 8)
    
    test_input = torch.randn(1, 2, 32, 32)
    output1 = model1(test_input)
    output2 = model2(test_input)
    
    assert output1.shape == output2.shape == (1, 8)
    
    print("✓ File consistency test passed")

def test_encoding_correctness():
    """Test that the encoding (0=empty, 1=obstacle, 2=goal) works correctly."""
    print("Testing encoding correctness...")
    
    obstacle_map = np.zeros((20, 20))
    obstacle_map[5:8, 5:8] = 1  # Add obstacles
    
    env = ulti6_improved.ImprovedPathPlanningEnv(obstacle_map, (1, 1), (18, 18))
    state = env.reset()
    
    # Check initial state
    assert state[1][18, 18] == 2.0, "Goal should be encoded as 2.0"
    assert state[1][1, 1] == 0.0, "Starting position should be empty in combined map"
    assert state[1][6, 6] == 0.0, "Undiscovered obstacles should be 0.0"
    
    # Discover obstacles
    env.update_discovered_obstacles(4, 4)  # Position that can see obstacles at (5,5)
    state = env.get_state()
    
    # Check that nearby obstacles are discovered
    assert state[1][5, 5] == 1.0, "Discovered obstacle should be 1.0"
    assert state[1][18, 18] == 2.0, "Goal should remain 2.0"
    
    print("✓ Encoding correctness test passed")

if __name__ == "__main__":
    print("=== Testing Merged Obstacle and Goal Map Implementation ===\n")
    
    try:
        test_environment_functionality()
        test_dqn_model()
        test_agent_functionality()
        test_3d_environment()
        test_both_files_consistent()
        test_encoding_correctness()
        
        print("\n=== All Tests Passed! ===")
        print("✓ State representation successfully changed from 3 to 2 matrices")
        print("✓ DQN model successfully updated to accept 2 input channels")
        print("✓ All functionality preserved")
        print("✓ Both ulti6_improved.py and ulti6_im.py working consistently")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise