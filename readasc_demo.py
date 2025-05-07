import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop


def read_asc_file(filename):
    metadata_keys = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value']
    metadata = {key: None for key in metadata_keys}

    with open(filename, 'r') as f:
        lines_read = 0
        while lines_read < 100:  # 最多读取100行头部
            line = f.readline().strip().lower()
            if not line or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            if key in metadata_keys:
                # 处理不同字段类型
                if key in ['ncols', 'nrows']:
                    metadata[key] = int(parts[1])
                else:
                    metadata[key] = float(parts[1])
                lines_read += 1
            if all(metadata.values()):
                break
        data = np.loadtxt(f, dtype=np.float32)
    missing = [k for k, v in metadata.items() if v is None]
    if missing:
        raise ValueError(f"缺少必要元数据字段: {missing}")

    return data, metadata


def create_obstacle_map(data, metadata):

    nodata = metadata['nodata_value']
    obstacle_map = np.where((data == nodata) | (data > 0), 1, 0)
    return obstacle_map.astype(np.int8)


def grid_to_geo(row, col, metadata):

    x = metadata['xllcorner'] + col * metadata['cellsize']
    y = metadata['yllcorner'] + (metadata['nrows'] - row - 1) * metadata['cellsize']
    return (x, y)

def astar(grid, start, end):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: abs(start[0] - end[0]) + abs(start[1] - end[1])}
    open_heap = []
    heappush(open_heap, (fscore[start], start))

    while open_heap:
        current = heappop(open_heap)[1]
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
                tentative_g = gscore[current] + 1
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    heappush(open_heap, (fscore[neighbor], neighbor))
    return None

def plot_2d_path(obstacle_map, paths, metadata, colors):
    plt.figure(figsize=(8, 6))
    plt.imshow(obstacle_map, cmap='gray_r', origin='upper')

    for idx, (path, start, end) in enumerate(paths):
        if path:
            y, x = zip(*path)
            plt.plot(x, y, color=colors[idx], linewidth=2, label=f'Path {idx + 1}')
        plt.scatter(start[1], start[0], color=colors[idx], marker='o', s=100)
        plt.scatter(end[1], end[0], color=colors[idx], marker='x', s=100)

    plt.title('2D Path Planning')
    plt.legend()
    plt.show()

def plot_3d_path(data, obstacle_map, paths, metadata, colors):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    data_clean = np.where(data == metadata['nodata_value'], np.nan, data)
    rows, cols = np.indices(data.shape)
    XX = metadata['xllcorner'] + cols * metadata['cellsize']
    YY = metadata['yllcorner'] + (metadata['nrows'] - rows - 1) * metadata['cellsize']
    surf = ax.plot_surface(XX, YY, data_clean, cmap='terrain',
                               rstride=10, cstride=10, alpha=0.7)
    fig.colorbar(surf, shrink=0.5, label='Elevation (m)')

    for idx, (path, start, end) in enumerate(paths):
        if path:
            geo_path = np.array([[
                metadata['xllcorner'] + col * metadata['cellsize'],
                metadata['yllcorner'] + (metadata['nrows'] - row - 1) * metadata['cellsize'],
                data[row, col]
                ] for (row, col) in path])

            ax.plot(geo_path[:, 0], geo_path[:, 1], geo_path[:, 2],
                        color=colors[idx], linewidth=3)
            ax.scatter(*geo_path[0], color=colors[idx], marker='o', s=100)
            ax.scatter(*geo_path[-1], color=colors[idx], marker='x', s=100)

    ax.set_title('3D Terrain Visualization')
    ax.set_xlabel('X (m)');
    ax.set_ylabel('Y (m)');
    ax.set_zlabel('Elevation (m)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        print('读取ASC文件...')
        data, metadata = read_asc_file('output_dem.asc')
        obstacle_map = create_obstacle_map(data, metadata)
        colors = ['red', 'blue', 'green', 'purple', 'orange']

        od_pairs = [((100, 200), (1500, 700))]  # 可扩展多个OD对
        paths = []
        for start, end in od_pairs:
            path = astar(obstacle_map, start, end)
            paths.append((path, start, end) if path else (None, start, end))
        print('可视化...')
        plot_2d_path(obstacle_map, paths, metadata, colors)
        plot_3d_path(data, obstacle_map, paths, metadata, colors)

    except Exception as e:
        print(f"程序异常: {str(e)}")
        print("可能原因：①文件路径错误 ②数据格式异常 ③硬件资源不足")