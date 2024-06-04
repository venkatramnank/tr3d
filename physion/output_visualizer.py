import os
import sys
import open3d as o3d
import numpy as np

def visualize_from_files(input_dir):
    # Load point cloud
    pcd_file = os.path.join(input_dir, f"{input_dir.split('/')[-1]}_pcd.pcd")
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Create Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point cloud to visualization
    vis.add_geometry(pcd)

    # Load and add bounding box line sets
    # bbox_files = [f for f in os.listdir(input_dir) if f.endswith("lineset.ply")]
    # for bbox_file in bbox_files:
    #     bbox_lineset = o3d.io.read_line_set(os.path.join(input_dir, bbox_file))
    #     vis.add_geometry(bbox_lineset)
    bbox_list_np = np.load(os.path.join(input_dir,f"{input_dir.split('/')[-1]}_bbox_list.npy"))
    bbox_list = list(bbox_list_np)
    for gt_bbox_info in (bbox_list):
        lines = [[0, 1], [1, 3], [2, 3], [2, 0],
             [4, 5], [5, 7], [6, 7], [6, 4],
             [1, 5], [0, 4], [2, 6], [3, 7]]

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(gt_bbox_info)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector([[0,1,0]] * len(lines))
        bbox = lineset 
        vis.add_geometry(bbox)
    
    # Set view control
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, -1, 0])
    vis.get_view_control().set_lookat([1, 1, 1])

    # Run visualization
    vis.run()
    vis.destroy_window()
    
def visulize_corners_boxes(corners_list_np):

    # Create Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Load and add bounding box line sets
    # bbox_files = [f for f in os.listdir(input_dir) if f.endswith("lineset.ply")]
    # for bbox_file in bbox_files:
    #     bbox_lineset = o3d.io.read_line_set(os.path.join(input_dir, bbox_file))
    #     vis.add_geometry(bbox_lineset)
    bbox_list = list(corners_list_np)
    for gt_bbox_info in (bbox_list):
        lines = [[0, 1], [1, 3], [2, 3], [2, 0],
             [4, 5], [5, 7], [6, 7], [6, 4],
             [1, 5], [0, 4], [2, 6], [3, 7]]

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(gt_bbox_info)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector([[0,1,0]] * len(lines))
        bbox = lineset 
        vis.add_geometry(bbox)
    
    # Set view control
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, -1, 0])
    vis.get_view_control().set_lookat([1, 1, 1])

    # Run visualization
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    visualize_from_files(input_dir)
