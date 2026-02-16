# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from django.conf import settings
# import os, cv2, tempfile, numpy as np, json, requests
# from deepface import DeepFace
# import mediapipe as mp
# from django.shortcuts import render
# from embeddings_gen import save_user_embeddings, validate_and_trim_video

# # Initialize Mediapipe
# face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.8)

# # Ensure embeddings directory exists
# EMBEDDINGS_DIR = os.path.join(settings.MEDIA_ROOT, 'embeddings')
# os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


# class GenerateUserEmbeddingsViewForm(APIView):
#     permission_classes = [AllowAny]

#     def get(self, request):
#         return render(request, "register_form.html")


#     def post(self, request):
#         print("📩 Incoming request keys:", request.data.keys())

#         try:
#             # 1️⃣ Extract the maid info JSON
#             raw_data = request.data.get('data')
#             if not raw_data:
#                 return Response({"error": "Missing data payload"}, status=400)

#             if isinstance(raw_data, str):
#                 data_json = json.loads(raw_data)
#             else:
#                 data_json = raw_data

#             if 'data' not in data_json or not isinstance(data_json['data'], list):
#                 return Response({"error": "Invalid JSON structure"}, status=400)

#             maid_info = data_json['data'][0]  # first record only
#             print("✅ Maid Info:", maid_info)

#             maid_id = maid_info.get("id")
#             uniqueId = maid_info.get("uniqueId")
#             maid_name = maid_info.get("maidName")
#             maid_mobile = maid_info.get("maidMobile")

#             if not all([maid_id, uniqueId]):
#                 return Response({"error": "Missing maid_id or uniqueId"}, status=400)

#             # 2️⃣ Get uploaded video from app
#             maid_video = request.FILES.get("maid_video")
#             if not maid_video:
#                 return Response({"error": "No maid video uploaded"}, status=400)

#             # Save temporary file
#             temp_path = os.path.join(tempfile.gettempdir(), f"maid_{maid_id}.mp4")
#             with open(temp_path, "wb") as f:
#                 for chunk in maid_video.chunks():
#                     f.write(chunk)

#             print("📁 Video temporarily saved at:", temp_path)

#             # 3️⃣ Validate + extract frames
#             valid, frames = validate_and_trim_video(temp_path)
#             os.remove(temp_path)

#             if not valid or not frames:
#                 return Response({"error": "No valid frames found in video"}, status=400)

#             embeddings_saved = 0

#             # 4️⃣ Generate embeddings per detected face
#             for i, frame in enumerate(frames):
#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 result = face_detection.process(rgb)
#                 if result.detections:
#                     for det in result.detections:
#                         ih, iw, _ = frame.shape
#                         box = det.location_data.relative_bounding_box
#                         x, y, w, h = (
#                             int(box.xmin * iw),
#                             int(box.ymin * ih),
#                             int(box.width * iw),
#                             int(box.height * ih)
#                         )
#                         x, y = max(0, x), max(0, y)
#                         crop = frame[y:min(y+h, ih), x:min(x+w, iw)]

#                         rep = DeepFace.represent(crop, model_name='Facenet', enforce_detection=False)
#                         if rep and isinstance(rep, list):
#                             for emb in rep:
#                                 save_user_embeddings(uniqueId, maid_mobile, emb['embedding'], i+1)
#                                 embeddings_saved += 1

#             if embeddings_saved == 0:
#                 return Response({"error": "No faces detected. Embeddings not generated."}, status=400)

#             # 5️⃣ Log to FastAPI (optional)
#             try:
#                 log_payload = {
#                     "user_id": int(maid_id),
#                     "unique_id": uniqueId,
#                     "maid_name": maid_name,
#                     "maid_mobile": maid_mobile,
#                     "status": "Success",
#                 }
#                 fastapi_logger_url = "http://localhost:8001/log-registration/"
#                 response = requests.post(fastapi_logger_url, json=log_payload)
#                 if response.status_code != 200:
#                     print("[Logger Warning] Failed to log registration:", response.text)
#             except Exception as log_error:
#                 print("[Logger Exception]", log_error)

#             # 6️⃣ Final success response
#             return Response({
#                 "message": f"✅ Embeddings generated successfully for {maid_name}",
#                 "maid_id": maid_id,
#                 "uniqueId": uniqueId,
#                 "embeddings_saved": embeddings_saved
#             }, status=200)

#         except Exception as e:
#             print("❌ Exception in GenerateUserEmbeddingsViewForm:", e)
#             return Response({"error": str(e)}, status=500)






from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.conf import settings
import os, cv2, tempfile, numpy as np, json, requests
from deepface import DeepFace
import mediapipe as mp
from django.shortcuts import render
from .embeddings_gen import save_user_embeddings, validate_and_trim_video

# Initialize Mediapipe
face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.8)

# Ensure embeddings directory exists
EMBEDDINGS_DIR = os.path.join(settings.MEDIA_ROOT, 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


class GenerateUserEmbeddingsViewForm(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return render(request, "register_form.html")


    def post(self, request):
        print("📩 Incoming request keys:", request.data.keys())

        try:
            # 1️⃣ Extract the maid info JSON
            raw_data = request.data.get('data')
            server_name = request.data.get("serverName")

            if not raw_data:
                return Response({"error": "Missing data payload"}, status=400)

            if not server_name:
                return Response({"error": "Missing serverName"}, status=400)
            
            server_name = str(server_name).strip().lower()


            if isinstance(raw_data, str):
                data_json = json.loads(raw_data)
            else:
                data_json = raw_data

            if 'data' not in data_json or not isinstance(data_json['data'], list):
                return Response({"error": "Invalid JSON structure"}, status=400)

            maid_info = data_json['data'][0]  # first record only
            print("✅ Maid Info:", maid_info)

            maid_id = maid_info.get("id")
            uniqueId = maid_info.get("uniqueId")
            maid_name = maid_info.get("maidName")
            maid_mobile = maid_info.get("maidMobile")

            if not all([maid_id, uniqueId]):
                return Response({"error": "Missing maid_id or uniqueId"}, status=400)

            # 2️⃣ Get uploaded video from app
            maid_video = request.FILES.get("maid_video")
            if not maid_video:
                return Response({"error": "No maid video uploaded"}, status=400)

            # Save temporary file
            temp_path = os.path.join(tempfile.gettempdir(), f"maid_{maid_id}.mp4")
            with open(temp_path, "wb") as f:
                for chunk in maid_video.chunks():
                    f.write(chunk)

            print("📁 Video temporarily saved at:", temp_path)

            # 3️⃣ Validate + extract frames
            valid, frames = validate_and_trim_video(temp_path)
            os.remove(temp_path)

            if not valid or not frames:
                return Response({"error": "No valid frames found in video"}, status=400)

            embeddings_saved = 0

            # 4️⃣ Generate embeddings per detected face
            for i, frame in enumerate(frames):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_detection.process(rgb)
                if result.detections:
                    for det in result.detections:
                        ih, iw, _ = frame.shape
                        box = det.location_data.relative_bounding_box
                        x, y, w, h = (
                            int(box.xmin * iw),
                            int(box.ymin * ih),
                            int(box.width * iw),
                            int(box.height * ih)
                        )
                        x, y = max(0, x), max(0, y)
                        crop = frame[y:min(y+h, ih), x:min(x+w, iw)]

                        rep = DeepFace.represent(crop, model_name='Facenet', enforce_detection=False)
                        if rep and isinstance(rep, list):
                            for emb in rep:
                                save_user_embeddings(server_name, uniqueId, maid_mobile, emb['embedding'], i+1)
                                embeddings_saved += 1

            if embeddings_saved == 0:
                return Response({"error": "No faces detected. Embeddings not generated."}, status=400)

            # 5️⃣ Log to FastAPI (optional)
            # try:
            #     log_payload = {
            #         "user_id": int(maid_id),
            #         "unique_id": uniqueId,
            #         "maid_name": maid_name,
            #         "maid_mobile": maid_mobile,
            #         "server_name": server_name,
            #         "status": "Success",
            #     }
            #     fastapi_logger_url = "http://localhost:8001/log-registration/"
            #     response = requests.post(fastapi_logger_url, json=log_payload)
            #     if response.status_code != 200:
            #         print("[Logger Warning] Failed to log registration:", response.text)
            # except Exception as log_error:
            #     print("[Logger Exception]", log_error)

            # 6️⃣ Final success response
            return Response({
                "message": f"✅ Embeddings generated successfully for {maid_name}",
                "maid_id": maid_id,
                "uniqueId": uniqueId,
                "serverName": server_name,
                "embeddings_saved": embeddings_saved
            }, status=200)

        except Exception as e:
            print("❌ Exception in GenerateUserEmbeddingsViewForm:", e)
            return Response({"error": str(e)}, status=500)
