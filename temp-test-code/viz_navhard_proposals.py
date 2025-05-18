import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def vis(data):
    vocab_size = data.shape[0]
    fig, ax = plt.subplots()
    for i in range(vocab_size):
        ax.plot(data[i, :, 0], data[i, :, 1])

    ax.legend()
    plt.show()


def vis_pdm(data, pdm, token, cam_path):
    scores = pdm[token]
    for m, v in scores.items():
        mask = v > 0.95
        vocab_size = data.shape[0]

        # Create a figure with two subplots: one for the image, one for the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Load and display the image on the left subplot
        img = mpimg.imread(cam_path)
        ax1.imshow(img)
        ax1.axis('off')  # Hide the axes for the image

        reds = []
        for i in range(vocab_size):
            if mask[i]:
                reds.append(data[i])
                # ax2.plot(data[i, :, 0], data[i, :, 1], 'r', alpha=1.0)
            else:
                ax2.plot(data[i, :, 0], data[i, :, 1], 'k', alpha=0.1)

        for red in reds:
            ax2.plot(red[:, 0], red[:, 1], 'r', alpha=1.0)

        ax2.legend()
        plt.title(m)
        plt.show()


def vis_proposals(token, proposals, cam_path):
    num_proposals = proposals.shape[0]

    # Create a figure with two subplots: one for the image, one for the trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Load and display the image on the left subplot
    img = mpimg.imread(cam_path)
    ax1.imshow(img)
    ax1.axis('off')  # Hide the axes for the image
    ax1.set_title(token)

    # Generate a color map for the proposals
    colors = plt.cm.rainbow(np.linspace(0, 1, num_proposals))

    # Plot each proposal with a different color
    for i in range(num_proposals):
        ax2.plot(proposals[i, :, 0], proposals[i, :, 1],
                 color=colors[i],
                 alpha=0.8,
                 label=f'Proposal {i + 1}')

    # Set equal aspect ratio for the trajectory plot
    ax2.set_aspect('equal', adjustable='datalim')
    # ax2.grid(True)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_title('Waypoint Trajectory Proposals')
    # ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    log_root = '/mnt/g/navsim/navhard_two_stage/openscene_meta_datas'
    all_logs = []
    for log_path in tqdm.tqdm(os.listdir(log_root)):
        curr_logs = pickle.load(open(f'{log_root}/{log_path}', 'rb'))
        all_logs += curr_logs
    log_root_syn = '/mnt/g/navsim/navhard_two_stage/synthetic_scene_pickles'

    for log_path in tqdm.tqdm(os.listdir(log_root_syn)):
        curr_logs = pickle.load(open(f'{log_root_syn}/{log_path}', 'rb'))['frames']
        all_logs += curr_logs

    logs = {}
    for log in all_logs:
        logs[log['token']] = log
    all_proposals = pickle.load(open('/mnt/f/e2e/navsim2/debug/navhard/traj_v2_bev_epoch41_n100_navhard.pkl', 'rb'))

    # Extract tokens from the CSV
    for token in set(all_proposals.keys()) & set(logs.keys()):
        # an np array of shape [Num_proposals, timestamps, xy coordinates]
        proposals = all_proposals[token]['proposals'][..., :2]
        if 'cams' in logs[token]:
            path = f"/mnt/g/navsim/navhard_two_stage/sensor_blobs/{logs[token]['cams']['CAM_F0']['data_path']}"
        else:
            path = f"/mnt/g/navsim/navhard_two_stage/sensor_blobs/{str(logs[token]['camera_dict']['cam_f0']['data_path'])}"

        vis_proposals(
            token,
            proposals,
            path
        )
