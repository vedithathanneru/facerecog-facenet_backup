import os
import cv2
import csv
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render
from datetime import datetime
import threading

# -------------------- THREAD LOCKS --------------------
csv_lock = threading.Lock()
embedding_lock = threading.Lock()
model_lock = threading.Lock()

# -------------------- LOAD MODEL ONCE --------------------
print("⚙️ Loading Facenet model once...")
DeepFace.build_model("Facenet")  # ❗this loads and caches model internally
print("✅ Facenet model ready.")

# -------------------- LOG FILE --------------------
LOGS_DIR = os.path.join(settings.BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "attendance_logs.csv")
os.makedirs(LOGS_DIR, exist_ok=True)


@csrf_exempt
def recognize_from_form(request):

    if request.method == "GET":
        return render(request, "recognise_form.html")

    elif request.method == "POST":
        try:
            form_data = request.POST
            file_data = request.FILES

            print("📦 FORM KEYS:", list(form_data.keys()))
            print("📸 FILE KEYS:", list(file_data.keys()))


            maid_mobile = form_data.get("groundmobiledispreq", "")
            maid_name = form_data.get("groundnamedispreq", "")
            unique_id = form_data.get("groundtemauniqueId", "")
            maid_location = form_data.get("groundtemalocation", "")
            shift_details = form_data.get("groundshiftdetials", "")
            temperature = form_data.get("groundtemp", "")
            mask_status = form_data.get("groundmask", "")
            device_name = form_data.get("indevicename", "")
            device_brand = form_data.get("indevicebrand", "")
            system_name = form_data.get("insystemname", "")
            ip_address = form_data.get("inipaddress", "")
            server_name = form_data.get("serverName") 
            image_file = file_data.get("PaymaaUpload1")

            # ✅ Step 2: Basic validations
            if not image_file:
                return JsonResponse({"status": "error", "message": "Image missing"}, status=400)
            if not unique_id:
                return JsonResponse({"status": "error", "message": "Missing unique ID"}, status=400)
            if not maid_mobile:
                return JsonResponse({"status": "error", "message": "Missing maid mobile"}, status=400)
            if not server_name or str(server_name).strip() == "" or server_name.lower() == "null":
                return JsonResponse({"status": "error", "message": "Invalid or missing server name"}, status=400)
            
            # ✅ Step 3: Parse location
            latitude, longitude = None, None
            if "&&" in maid_location:
                parts = maid_location.split("&&")
                if len(parts) == 2:
                    latitude, longitude = parts
            print(f"🗺 Parsed Location → Lat: {latitude}, Lon: {longitude}")

            # ✅ Step 4: Convert image to OpenCV format
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                return JsonResponse({"status": "error", "message": "Invalid image"}, status=400)

            print("✅ Image decoded successfully. Shape:", image.shape)


            # -------------------- Generate Embedding --------------------
            print("🧠 Extracting embedding using DeepFace (Facenet)...")
            with model_lock:
                embedding_result = DeepFace.represent(
                    image,
                    model_name='Facenet',
                    enforce_detection=False
                )

            embedding = embedding_result[0]['embedding']
            print("✅ Embedding extracted successfully")


            # -------------------- Verification --------------------
            verified, weighted_sum = verify_employee_identity(
                maid_mobile,
                unique_id,
                embedding,
                server_name
            )

            # -------------------- Logging --------------------
            log_to_csv({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "maid_name": maid_name,
                "maid_mobile": maid_mobile,
                "unique_id": unique_id,
                "verified": verified,
                "weighted_sum": round(weighted_sum, 4),
                "mask_status": mask_status or "N/A",
                "temperature": temperature or "N/A",
                "shift_details": shift_details or "N/A",
                "latitude": latitude or "N/A",
                "longitude": longitude or "N/A",
                "device_name": device_name or "N/A",
                "device_brand": device_brand or "N/A",
                "system_name": system_name or "N/A",
                "ip_address": ip_address or "N/A",
                "server_name": server_name or "N/A"
            })

            print("🧾 Log saved to CSV file successfully.")
            print("===================== PROCESS COMPLETED =====================\n")


            return JsonResponse({
                "status": "verified" if verified else "not_verified",
                "maid_name": maid_name,
                "mobile": maid_mobile,
                "uniqueId": unique_id,
                "weighted_sum": weighted_sum
            })

        except Exception as e:
            print("❌ ERROR:", e)
            return JsonResponse({"status": "error", "message": str(e)}, status=500)


# -------------------- VERIFY IDENTITY --------------------
def verify_employee_identity(user_id, uniqueId, uploaded_embedding, server_name):

    print("\n[VERIFY] Starting identity verification...")

    # ✅ Validate server name
    if not server_name or str(server_name).strip().lower() == "null":
        print("❌ Invalid server_name provided")
        return False, 0.0
    
    # ✅ Build paths
    server_folder = os.path.join(settings.MEDIA_ROOT, 'embeddings', str(server_name).strip().lower())
    user_folder = os.path.join(server_folder, f"company_{uniqueId}")

    print(f"[VERIFY] Checking folder: {user_folder}")

    if not os.path.exists(user_folder):
        print(f"❌ Folder not found: {user_folder}")
        return False, 0.0

    user_files = [
        f for f in os.listdir(user_folder)
        if f.startswith(f"{user_id}_") and f.endswith('.npy')
    ]

    print(f"[VERIFY] Found {len(user_files)} embeddings for user {user_id}")


    if not user_files:
        return False, 0.0

    weighted_sum = 0.0

    with embedding_lock:
        for f in user_files:
            file_path = os.path.join(user_folder, f)
            stored_embedding = np.load(file_path)

            distance = cosine(uploaded_embedding, stored_embedding)
            print(f"→ Compared with {f}: Distance={distance:.4f}")
            if distance < 0.5:
                similarity = 1 - distance
                weighted_sum += similarity ** 2

        # ✅ Threshold for verification
        verified = weighted_sum >= 4.3
        print(f"[VERIFY RESULT] Verified={verified}, Weighted Sum={weighted_sum:.4f}\n")


    return weighted_sum >= 4.5, weighted_sum


# -------------------- CSV LOGGING --------------------
def log_to_csv(data):
    with csv_lock:
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)


@csrf_exempt
def check_embedding_status(request):
    try:
        maidMobile = request.GET.get("maidMobile")
        uniqueId = request.GET.get("uniqueId")
        serverName = request.GET.get("serverName")

        print(serverName)

        # Basic validations
        if not maidMobile or not uniqueId or not serverName:
            return JsonResponse({"registered": False})
        
        # Build new embedding directory path
        server_folder = os.path.join(
            settings.MEDIA_ROOT,
            "embeddings",
            str(serverName).strip().lower()
        )

        embed_dir = os.path.join(server_folder, f"company_{uniqueId}")

        # Check folder existence
        if not os.path.exists(embed_dir):
            return JsonResponse({"registered": False})

        # embed_dir = os.path.join(settings.MEDIA_ROOT, "embeddings", f"company_{uniqueId}")

        files = [f for f in os.listdir(embed_dir) if f.startswith(f"{maidMobile}_") and f.endswith(".npy")]

        if len(files) == 0:
            return JsonResponse({"registered": False})

        return JsonResponse({"registered": True})

    except Exception:
        return JsonResponse({"registered": False})