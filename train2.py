from ulti7_im import *
import time

if __name__ == "__main__":
    try:
        # 1) Read data
        data, metadata = read_asc_file('output_dem4.asc')

        # 2) Create 2D obstacle map for standard approach
        obstacle_map = create_obstacle_map(data, metadata)

        # 3) Define start/end in geo coords, then convert to row/col
        start_geo = (metadata['xllcorner'] + 10 * metadata['cellsize'],
                     metadata['yllcorner'] + (metadata['nrows'] - 10 - 1) * metadata['cellsize'])
        end_geo = (metadata['xllcorner'] + 150 * metadata['cellsize'],
                   metadata['yllcorner'] + (metadata['nrows'] - 150 - 1) * metadata['cellsize'])
        start_row, start_col = geo_to_grid(*start_geo, metadata)
        end_row, end_col = geo_to_grid(*end_geo, metadata)

        delta_z = 1
        start_3d = (start_row, start_col, data[start_row, start_col] + 3)
        end_3d = (end_row, end_col, data[end_row, end_col] + 3)

        s = f'MODEL 🚀 torch {torch.__version__} '
        n = torch.cuda.device_count()
        space = ' ' * (len(s) + 1)
        for d in range(n):
            p = torch.cuda.get_device_properties(d)
            s += f"{'' if d == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
        print(s)


        # 4) Demonstration of standard DQN  3D
        def train_dqn_agent(is_3d=False, ep=50):
            print(f"\n{'=' * 50}")
            print(f"开始训练 {'3D' if is_3d else '2D'} DQN 智能体")
            print(f"训练回合数: {ep}")
            print(f"{'=' * 50}")
            # 记录开始时间
            start_time = time.time()
            if is_3d:
                env = PathPlanningEnv(obstacle_map, start_3d, end_3d,
                                      is_3d=True, elevation_data=data, delta_z=delta_z)
            else:
                env = PathPlanningEnv(obstacle_map, (start_row, start_col), (end_row, end_col))
            agent = DQNAgent(env, is_3d=is_3d)
            print(f"训练 {'3D' if is_3d else '2D'} DQN 智能体中...")
            # 训练并获取统计信息
            training_stats = agent.train(episodes=ep, layer_min=None, layer_max=None)
            # 记录结束时间
            end_time = time.time()
            training_time = end_time - start_time
            # 获取路径规划结果
            path_result = agent.get_path()
            # 计算路径长度
            path_length = calculate_path_length(path_result['path'], metadata, is_3d)
            if path_result['path']:
                if is_3d:
                    start_point = path_result['path'][0]
                    end_point = path_result['path'][-1]
                    print(f"起点坐标: (行={start_point[0]}, 列={start_point[1]}, 高度={start_point[2]:.2f}m)")
                    print(f"终点坐标: (行={end_point[0]}, 列={end_point[1]}, 高度={end_point[2]:.2f}m)")

                    # 新增：绘制高度变化图（仅3D）
                    heights = [p[2] for p in path_result['path']]
                    plt.figure(figsize=(10, 5))
                    plt.plot(heights, 'b-', linewidth=2)
                    plt.scatter(0, heights[0], c='g', s=100, label='start')
                    plt.scatter(len(heights) - 1, heights[-1], c='r', s=100, label='end')
                    plt.xlabel('step')
                    plt.ylabel('altitude (m)')
                    plt.title('3D DQN altitude with step')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                else:
                    start_point = path_result['path'][0]
                    end_point = path_result['path'][-1]
                    print(f"起点坐标: (行={start_point[0]}, 列={start_point[1]})")
                    print(f"终点坐标: (行={end_point[0]}, 列={end_point[1]})")
            else:
                print("警告: 未找到有效路径")
            # 打印训练统计信息
            print(f"\n{'=' * 50}")
            print(f"{'3D' if is_3d else '2D'} DQN 训练完成统计:")
            print(f"训练时间: {training_time:.2f} 秒 ({training_time / 60:.2f} 分钟)")
            print(f"路径规划成功率: {training_stats['success_rate']:.2f}%")
            print(f"最终路径长度: {path_length:.2f} 米")
            print(f"路径规划是否成功: {'是' if path_result['success'] else '否'}")
            print(f"最终距离目标: {path_result['final_distance']:.2f} 单位")
            print(f"{'=' * 50}\n")

            agent.plot_loss()
            agent.save_model(f'dqn_agent_{"3d" if is_3d else "2d"}.pth')
            return path_result


        dqn_3d_result = train_dqn_agent(is_3d=True, ep=100)
        all_paths = [
            ('DQN 3D', dqn_3d_result, True)]

        # 5) Layered approach demonstration
        layer_ranges = [(30, 40), (40, 50), (50, 60)]
        multi_layer_paths = []

        for (lmin, lmax) in layer_ranges:
            print(f"\n{'=' * 50}")
            print(f"开始训练分层 DQN: {lmin}-{lmax}m")
            print(f"{'=' * 50}")

            # 记录开始时间
            start_time = time.time()

            lm = create_layered_obstacle_map(data, metadata, lmin, lmax)
            layered_env = LayeredDQNEnv(lm, (start_row, start_col), (end_row, end_col))
            layered_agent = DQNAgent(layered_env, is_3d=False)

            # 训练并获取统计信息
            training_stats = layered_agent.train(episodes=100, layer_min=lmin, layer_max=lmax)

            # 记录结束时间
            end_time = time.time()
            training_time = end_time - start_time

            layered_agent.plot_loss()
            layered_result = layered_agent.get_path()

            # 计算路径长度
            path_length = calculate_path_length(layered_result['path'], metadata, is_3d=False)
            if layered_result['path']:
                start_point = layered_result['path'][0]
                end_point = layered_result['path'][-1]
                # 获取起点和终点的高度（地面高度+3米）
                start_height = data[start_point[0], start_point[1]] +  + 5
                end_height = data[end_point[0], end_point[1]] +  + 5


                print(
                    f"分层 {lmin}-{lmax}m 起点坐标: (行={start_point[0]}, 列={start_point[1]}, 高度={start_height:.2f}m)")
                print(f"分层 {lmin}-{lmax}m 终点坐标: (行={end_point[0]}, 列={end_point[1]}, 高度={end_height:.2f}m)")
            else:
                print(f"分层 {lmin}-{lmax}m 警告: 未找到有效路径")

            # 打印训练统计信息
            print(f"\n{'=' * 50}")
            print(f"分层 DQN ({lmin}-{lmax}m) 训练完成统计:")
            print(f"训练时间: {training_time:.2f} 秒 ({training_time / 60:.2f} 分钟)")
            print(f"路径规划成功率: {training_stats['success_rate']:.2f}%")
            print(f"最终路径长度: {path_length:.2f} 米")
            print(f"路径规划是否成功: {'是' if layered_result['success'] else '否'}")
            print(f"最终距离目标: {layered_result['final_distance']:.2f} 单位")
            print(f"{'=' * 50}\n")

            # 保存路径、层范围和success状态
            multi_layer_paths.append((
                layered_result['path'],
                (lmin, lmax),
                layered_result['success']
            ))
            all_paths.append((f'Layered DQN {lmin}-{lmax}m', layered_result, False))

        print("绘制所有方案对比图...")
        plot_comparison(obstacle_map, all_paths, metadata)

    except Exception as e:
        print(f"Error: {str(e)}")