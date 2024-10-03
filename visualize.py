import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_tints(base_colors, num_tints=3):
    def lighten_color(color, factor):
        color_rgb = mcolors.to_rgb(color)
        return [(1 - factor) * c + factor for c in color_rgb]

    tints = [[] for _ in range(num_tints)]
    for color in base_colors:
        for i in range(num_tints):
            tint_color = lighten_color(color, (i + 1) / (num_tints + 1))
            tints[i].append(mcolors.to_hex(tint_color))
    
    return tints

def extract_coordinates(list_of_lists):
    coordinates = []
    for sublist in list_of_lists:
        z = [coord[0] for coord in sublist]
        x = [coord[1] for coord in sublist]
        y = [coord[2] for coord in sublist]
        coordinates.append((z, x, y))
    return coordinates

def create_scatter3d_traces(coordinates, color_list):
    traces = []
    colors = color_list
    for i, (z, x, y) in enumerate(coordinates):
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=colors[i % len(colors)],
                opacity=1
            ),
            name=f'Segment {i+1}'
        )
        traces.append(trace)
    return traces

def visualize_skeleton(skeleton, tips, knots, output_path, title):
    # Prepare skeleton data for plotting
    image_stack = np.transpose(skeleton, (1, 2, 0))
    x, y, z = image_stack.shape
    Y, X, Z = np.meshgrid(np.arange(y), np.arange(x), np.arange(z))
    colors = image_stack.ravel()

    # Omit background (0-valued) voxels
    visible_mask = colors != 0

    # Extract coordinates for skeleton tips
    tips_z = [point[0] for point in tips]
    tips_x = [point[1] for point in tips]
    tips_y = [point[2] for point in tips]

    # Extract coordinates for skeleton knots
    knot_z = [point[0] for point in knots]
    knot_x = [point[1] for point in knots]
    knot_y = [point[2] for point in knots]

    # Plotly figure
    fig = go.Figure()

    # Skeleton tips
    fig.add_trace(go.Scatter3d(
        x=tips_x,
        y=tips_y,
        z=tips_z,
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=.9
        ),
        name='Tips'
    ))

    # Skeleton knots
    fig.add_trace(go.Scatter3d(
        x=knot_x,
        y=knot_y,
        z=knot_z,
        mode='markers',
        marker=dict(
            size=4,
            color='blue',
            opacity=.9
        ),
        name='Knots'
    ))

    # Skeleton structure
    fig.add_trace(go.Scatter3d(
        x=X.ravel()[visible_mask],
        y=Y.ravel()[visible_mask],
        z=Z.ravel()[visible_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='black',
            colorscale='Viridis',
            opacity=.1
        ),
        name='Skeleton'
    ))

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            zaxis_title='Z (image index)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=z/1000),  # Scale z axis to aspect ratio similar to ImageJ 3d Viewer
            zaxis=dict(range=[0, skeleton.shape[0]]),
            xaxis=dict(range=[0, skeleton.shape[1]]),
            yaxis=dict(range=[0, skeleton.shape[2]]),
        ),
        autosize=True
    )

    # Save the plot to an HTML file
    pio.write_html(fig, file=output_path, auto_open=False)

def visualize_segments(segments, skeleton, output_path, title):
    # Prepare skeleton data for plotting
    image_stack = np.transpose(skeleton, (1, 2, 0))
    x, y, z = image_stack.shape
    Y, X, Z = np.meshgrid(np.arange(y), np.arange(x), np.arange(z))
    colors = image_stack.ravel()
    visible_mask = colors != 0

    # Extract coordinates
    seg_coordinates = extract_coordinates(segments)

    # Set colors
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
    tints = create_tints(base_colors)
    color_list = base_colors + tints[0] + tints[1] + tints[2]

    # Create Scatter3d traces
    traces = create_scatter3d_traces(seg_coordinates, color_list)

    # Visualize segments
    fig = go.Figure(data=traces)

    # Original skeleton structure
    fig.add_trace(go.Scatter3d(
        x=X.ravel()[visible_mask],
        y=Y.ravel()[visible_mask],
        z=Z.ravel()[visible_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='black',
            colorscale='Viridis',
            opacity=.1
        ),
        name='Skeleton'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            zaxis_title='Z (image index)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=z/1000),
            zaxis=dict(range=[0, skeleton.shape[0]]),
            xaxis=dict(range=[0, skeleton.shape[1]]),
            yaxis=dict(range=[0, skeleton.shape[2]]),
        ),
        autosize=True
    )

    # Save the plot to an HTML file
    pio.write_html(fig, file=output_path, auto_open=False)

def visualize_matched_segments(matched_segments, skeletons, output_path, title):
    # Prepare skeleton data for plotting (using the first timepoint)
    image_stack = np.transpose(skeletons[0], (1, 2, 0))
    x, y, z = image_stack.shape
    Y, X, Z = np.meshgrid(np.arange(y), np.arange(x), np.arange(z))
    colors = image_stack.ravel()
    visible_mask = colors != 0

    # Extract coordinates for each timepoint
    seg_coordinates = [extract_coordinates(segments) for segments in matched_segments]

    # Set colors
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'magenta']  # Cyan omitted
    tints = create_tints(base_colors)
    color_lists = [base_colors, tints[0], tints[1], tints[2]]

    # Create Scatter3d traces for each timepoint
    all_traces = []
    for t, coordinates in enumerate(seg_coordinates):
        traces = create_scatter3d_traces(coordinates, color_lists[t])
        all_traces.extend(traces)

    # Visualize segments
    fig = go.Figure(data=all_traces)

    # Original skeleton structure (using the first timepoint)
    fig.add_trace(go.Scatter3d(
        x=X.ravel()[visible_mask],
        y=Y.ravel()[visible_mask],
        z=Z.ravel()[visible_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='black',
            colorscale='Viridis',
            opacity=.1
        ),
        name='Skeleton'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            zaxis_title='Z (image index)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=z/1000),
            zaxis=dict(range=[0, skeletons[0].shape[0]]),
            xaxis=dict(range=[0, skeletons[0].shape[1]]),
            yaxis=dict(range=[0, skeletons[0].shape[2]]),
        ),
        autosize=True
    )

    # Save the plot to an HTML file
    pio.write_html(fig, file=output_path, auto_open=False)

# Function to add colors to labeled tiff
def color_labeled_volume(labeled_volume, num_labels):
    np.random.seed(0)  # For consistent colors across runs
    colors = np.random.randint(0, 255, size=(num_labels + 1, 3), dtype=np.uint8)  # +1 for background
    colors[0] = [0, 0, 0]  # Set background color to black

    colored_volume = np.zeros((*labeled_volume.shape, 3), dtype=np.uint8)
    for i in range(num_labels + 1):
        mask = labeled_volume == i
        colored_volume[mask] = colors[i]

    return colored_volume

def visualize_mips(pvd, display=False, save=False, output_dir=''):
    plt.ioff()  # Turn off interactive mode
    fig = plt.figure(figsize=(12, 12))
    
    for ii in range(4):
        ax = fig.add_subplot(2, 2, ii+1)
        ax.imshow(pvd.mip[ii].astype(np.uint8)*255)
        ax.set_title(f'Timepoint {ii}')
        ax.axis('off')

    plt.tight_layout()

    if save:
        plt.savefig(f"{output_dir}timepoint_mips.png", dpi=150, bbox_inches='tight')
    
    if display:
        plt.ion() # Turn on interactive mode
        plt.show()
    
    plt.close(fig)