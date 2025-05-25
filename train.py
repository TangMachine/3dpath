from ulti import *

if __name__ == "__main__":
    try:
        # 1) Read data
        data, metadata = read_asc_file('output_dem.asc')

        # 2) Create 2D obstacle map for standard approach
        obstacle_map = create_obstacle_map(data, metadata)

        # 3) Define start/end in geo coords, then convert to row/col
        start_geo = (metadata['xllcorner'] + 200 * metadata['cellsize'],
                     metadata['yllcorner'] + (metadata['nrows'] - 300 - 1) * metadata['cellsize'])
        end_geo = (metadata['xllcorner'] + 700 * metadata['cellsize'],
                   metadata['yllcorner'] + (metadata['nrows'] - 1800 - 1) * metadata['cellsize'])
        start_row, start_col = geo_to_grid(*start_geo, metadata)
        end_row, end_col = geo_to_grid(*end_geo, metadata)

        delta_z = 1
        start_3d = (start_row, start_col, data[start_row, start_col] + 5)
        end_3d = (end_row, end_col, data[end_row, end_col] + 5)


        # 4) Demonstration of standard DQN  3D
        def train_dqn_agent(is_3d=False):
            if is_3d:
                env = PathPlanningEnv(obstacle_map, start_3d, end_3d,
                                      is_3d=True, elevation_data=data, delta_z=delta_z)
            else:
                env = PathPlanningEnv(obstacle_map, (start_row, start_col), (end_row, end_col))

            agent = DQNAgent(env, is_3d=is_3d)
            print(f"Training {'3D' if is_3d else '2D'} DQN agent...")
            agent.train(episodes=50,layer_min=None, layer_max=None)  # fewer episodes for demo
            agent.save_model(f'dqn_agent_{"3d" if is_3d else "2d"}.pth')
            return agent.get_path()

        dqn_3d_result = train_dqn_agent(is_3d=True)
        all_paths = [
            ('DQN 3D', dqn_3d_result, True)]

        # 5) Layered approach demonstration
        layer_ranges = [(30, 40), (40, 50), (50, 60)]
        multi_layer_paths = []
        for (lmin, lmax) in layer_ranges:
            print(f"Layered DQN: {lmin}-{lmax}m")
            lm = create_layered_obstacle_map(data, metadata, lmin, lmax)
            layered_env = LayeredDQNEnv(lm, (start_row, start_col), (end_row, end_col))
            layered_agent = DQNAgent(layered_env, is_3d=False)
            layered_agent.train(episodes=50,layer_min=lmin,layer_max=lmax)  # fewer episodes for demo
            layered_result = layered_agent.get_path()  # 获取完整结果
            # 保存路径、层范围和success状态
            multi_layer_paths.append((
                layered_result['path'],
                (lmin, lmax),
                layered_result['success']  # 新增success状态
            ))
            all_paths.append((f'Layered DQN {lmin}-{lmax}m', layered_result, False))



        print("绘制所有方案对比图...")
        plot_comparison(obstacle_map, all_paths, metadata)


    except Exception as e:
        print(f"Error: {str(e)}")