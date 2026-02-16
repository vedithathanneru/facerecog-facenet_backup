# import os, cv2
# import numpy as np
# from django.conf import settings

# EMBEDDINGS_DIR = os.path.join(settings.MEDIA_ROOT, 'embeddings')
# os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# def save_user_embeddings(uniqueId, person_id, embeddings, frame_index):
#     company_folder = os.path.join(EMBEDDINGS_DIR, f"company_{uniqueId}")
#     os.makedirs(company_folder, exist_ok=True)
#     embeddings_file = os.path.join(company_folder, f"{person_id}_{frame_index}.npy")
#     np.save(embeddings_file, np.array(embeddings))

# def validate_and_trim_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     # max_frames = fps * 5
#     max_frames = 15
#     frames = []
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count < max_frames:
#             frames.append(frame)
#         else:
#             break
#         frame_count += 1

#     cap.release()
#     return True, frames


import os
import cv2
import numpy as np
from django.conf import settings

EMBEDDINGS_DIR = os.path.join(settings.MEDIA_ROOT, 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def save_user_embeddings(server_name, uniqueId, person_id, embeddings, frame_index):

    """
    Save embeddings under:
    media/embeddings/{server_name}/company_{uniqueId}/person_frame.npy
    """

    server_name = str(server_name).strip().lower()

    # Create server-level folder
    server_folder = os.path.join(EMBEDDINGS_DIR, server_name)
    os.makedirs(server_folder, exist_ok=True)

    # Create company-level folder
    company_folder = os.path.join(server_folder, f"company_{uniqueId}")
    os.makedirs(company_folder, exist_ok=True)

    embeddings_file = os.path.join(company_folder, f"{person_id}_{frame_index}.npy")
    np.save(embeddings_file, np.array(embeddings))

def validate_and_trim_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # max_frames = fps * 5
    max_frames = 15
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count < max_frames:
            frames.append(frame)
        else:
            break
        frame_count += 1

    cap.release()

    if len(frames) == 0:
        return False, []
    
    return True, frames