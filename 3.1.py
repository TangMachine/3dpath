import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import csv

def read_asc_file(filename):
    metadata_keys = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value']
    metadata = {key: None for key in metadata_keys}

    with open(filename, 'r') as f:
        lines_read = 0
        while lines_read < 100:
            line = f.readline().strip().lower()
            if not line or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            if key in metadata_keys:
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
        raise ValueError(f"Missing metadata fields: {missing}")

    return data, metadata


def create_obstacle_map(data, metadata):
    nodata = metadata['nodata_value']
    obstacle_map = np.where((data == nodata) | (data > 0), 1, 0)
    return obstacle_map.astype(np.int8)


def grid_to_geo(row, col, metadata):
    x = metadata['xllcorner'] + col * metadata['cellsize']
    y = metadata['yllcorner'] + (metadata['nrows'] - row - 1) * metadata['cellsize']
    return (x, y)


def geo_to_grid(x, y, metadata):
    col = int((x - metadata['xllcorner']) // metadata['cellsize'])
    row = int(metadata['nrows'] - (y - metadata['yllcorner']) // metadata['cellsize'] - 1)
    if row < 0 or row >= metadata['nrows'] or col < 0 or col >= metadata['ncols']:
        raise ValueError("Coordinates out of bounds")
    return (row, col)


def astar_2d(grid, start, end):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    closed_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: abs(start[0] - end[0]) + abs(start[1] - end[1])}
    heap = []
    heappush(heap, (fscore[start], start))

    while heap:
        _, current = heappop(heap)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)
        for dx, dy in neighbors:
            nr, nc = current[0] + dx, current[1] + dy
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr, nc] == 1:
                    continue
                tentative_g = gscore[current] + 1
                if (nr, nc) in closed_set and tentative_g >= gscore.get((nr, nc), float('inf')):
                    continue
                if tentative_g < gscore.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    gscore[(nr, nc)] = tentative_g
                    fscore[(nr, nc)] = tentative_g + abs(nr - end[0]) + abs(nc - end[1])
                    heappush(heap, (fscore[(nr, nc)], (nr, nc)))
    return None


def astar_3d(data, metadata, start, end, delta_z):
    def heuristic(a, b):
        dx = (a[1] - b[1]) * metadata['cellsize']
        dy = (a[0] - b[0]) * metadata['cellsize']
        dz = a[2] - b[2]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    open_heap = []
    gscore = {start: 0}
    fscore = {start: heuristic(start, end)}
    heappush(open_heap, (fscore[start], gscore[start], start))  # (f, g, node)
    came_from = {}

    while open_heap:
        current_f, current_g, current = heappop(open_heap)

        # Lazy A*: Skip if this node has been improved
        if current_g > gscore.get(current, float('inf')):
            continue

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        # Generate neighbors with simplified cost calculation
        neighbors = []
        # Horizontal moves (4 directions)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row = current[0] + dx
            new_col = current[1] + dy
            if 0 <= new_row < data.shape[0] and 0 <= new_col < data.shape[1]:
                # Allow movement only if terrain height <= current z
                if data[new_row, new_col] <= current[2]:
                    neighbors.append(((new_row, new_col, current[2]), metadata['cellsize']))

        # Vertical moves (up/down)
        for dz in [delta_z, -delta_z]:
            new_z = current[2] + dz
            # Ensure altitude doesn't go below terrain
            if new_z >= data[current[0], current[1]]:
                neighbors.append(((current[0], current[1], new_z), abs(dz)))

        for neighbor, move_cost in neighbors:
            tentative_g = current_g + move_cost
            # Update if this path is better
            if tentative_g < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore_neighbor = tentative_g + heuristic(neighbor, end)
                heappush(open_heap, (fscore_neighbor, tentative_g, neighbor))

    return None


def plot_comparison(obstacle_map, paths_2d, paths_3d, metadata):
    plt.figure(figsize=(12, 8))
    plt.imshow(obstacle_map, cmap='gray_r', origin='upper')

    # Plot 2D paths
    for path, start, end in paths_2d:
        if path:
            y, x = zip(*path)
            plt.plot(x, y, 'r-', linewidth=2, label='2D Path')

    # Plot 3D paths
    for path, start, end in paths_3d:
        if path:
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            plt.plot(cols, rows, 'b--', linewidth=2, label='3D Path')
            # Mark elevation changes
            prev_z = None
            for p in path:
                if prev_z is not None and p[2] != prev_z:
                    plt.scatter(p[1], p[0], c='g', s=50, zorder=3)
                prev_z = p[2]

    plt.legend()
    plt.title('2D vs 3D Path Comparison')
    plt.show()


def generate_table(path, metadata, data, is_3d=False):
    table = []
    total_dist = 0.0
    prev_geo = None
    for step, point in enumerate(path):
        row, col = point[0], point[1]
        x, y = grid_to_geo(row, col, metadata)
        z = point[2] if is_3d else data[row, col]
        h = data[row, col]

        if step > 0:
            dx = x - prev_geo[0]
            dy = y - prev_geo[1]
            dz = z - prev_geo[2] if is_3d else 0
            dist = np.sqrt(dx ** 2 + dy ** 2 + (dz ** 2 if is_3d else 0))
            total_dist += dist

        table.append({
            'Step': step + 1,
            'X': x,
            'Y': y,
            'Z': z,
            'Terrain Height': h,
            'Cumulative Distance': total_dist
        })
        prev_geo = (x, y, z) if is_3d else (x, y, data[row, col])

    return table


def export_csv(table, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    try:
        data, metadata = read_asc_file('output_dem.asc')
        obstacle_map = create_obstacle_map(data, metadata)

        # Define OD pairs in geographic coordinates
        start_geo = (metadata['xllcorner'] + 100 * metadata['cellsize'],
                     metadata['yllcorner'] + (metadata['nrows'] - 200 - 1) * metadata['cellsize'])
        end_geo = (metadata['xllcorner'] + 500 * metadata['cellsize'],
                   metadata['yllcorner'] + (metadata['nrows'] - 400 - 1) * metadata['cellsize'])

        # Convert to grid coordinates
        start_row, start_col = geo_to_grid(*start_geo, metadata)
        end_row, end_col = geo_to_grid(*end_geo, metadata)
        print(f"Start: {start_row}, {start_col} ")
        # 2D Path
        print(1)
        path_2d = astar_2d(obstacle_map, (start_row, start_col), (end_row, end_col))

        # 3D Path with optimized parameters
        print(2)
        delta_z  =6
        start_3d = (start_row, start_col, data[start_row, start_col] + delta_z)
        end_3d = (end_row, end_col, data[end_row, end_col] + delta_z)
        path_3d = astar_3d(data, metadata, start_3d, end_3d, delta_z)  # 增大步长提升速度

        # Generate tables
        if path_2d:
            print(3)
            table_2d = generate_table(path_2d, metadata, data)
            export_csv(table_2d, '2d_path.csv')
        if path_3d:
            print(4)
            table_3d = generate_table(path_3d, metadata, data, is_3d=True)
            export_csv(table_3d, '3d_path.csv')

        # Visualization
        print(5)
        plot_comparison(obstacle_map,
                        [(path_2d, (start_row, start_col), (end_row, end_col))] if path_2d else [],
                        [(path_3d, start_3d, end_3d)] if path_3d else [],
                        metadata)

    except Exception as e:
        print(f"Error: {str(e)}")