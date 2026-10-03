# Setup, Deployment & Mobile HTTPS Configuration

## 1. Prerequisites
- **Computer/Laptop:** Python 3.10+, Node.js 18+, Git.
- **Smartphone:** iOS (Safari 15+) or Android (Chrome 100+).
- **Physical Hardware:** 3D-printed optical spacer (matching phone camera dimensions) with contact reference patch.
- **Network Tunnel Tool:** Cloudflare Tunnel (`cloudflared`) or `ngrok` (required for mobile HTTPS camera permissions).

---

## 2. Why HTTPS is Mandatory for Mobile Devices
Modern mobile browsers (Chrome on Android and Safari on iOS) **strictly enforce HTTPS** for hardware camera access (`navigator.mediaDevices.getUserMedia`). 
While desktop browsers allow unencrypted camera access on `http://localhost`, your phone connecting over your local network (e.g. `http://192.168.1.5:5173`) will **refuse camera permissions**.

To connect your physical smartphone camera to your development server, you must expose an HTTPS endpoint.

---

## 3. Recommended Development Ingress: Cloudflare Tunnel (Free & Instant)

Cloudflare Tunnel provides a secure, encrypted HTTPS URL directly to your local laptop without opening router ports or buying SSL certificates:

1. **Install Cloudflare Tunnel:**
   - Windows (via winget or direct binary):
     ```powershell
     winget install Cloudflare.cloudflared
     ```
   - macOS / Linux:
     ```bash
     brew install cloudflared   # or apt-get install cloudflared
     ```
2. **Start Frontend & Backend on Laptop:**
   ```bash
   # Terminal 1: Backend
   cd backend && uvicorn app.main:app --reload --port 8000

   # Terminal 2: Frontend
   cd frontend && npm run dev -- --host 0.0.0.0 --port 5173
   ```
3. **Expose Frontend over HTTPS:**
   ```bash
   # Terminal 3: Tunnel to Vite Dev Server
   cloudflared tunnel --url http://localhost:5173
   ```
   *Cloudflare outputs a public URL: `https://xxxx-xxxx.trycloudflare.com`.*
4. **Open the URL on Your Smartphone:**
   - Scan the terminal QR code or open the `https://...` link in mobile Chrome/Safari.
   - The browser will ask for camera permission and open the rear camera feed immediately!

*(Alternative: `ngrok http 5173` achieves the same result).*

---

## 4. Phone & 3D Spacer Setup Protocol

1. **Lens Alignment:**
   - Slide the phone into the 3D-printed cradle. Ensure the primary rear camera lens is perfectly concentric with the spacer cone aperture.
   - If using a multi-lens phone (e.g., iPhone 14/15 Pro, Samsung S23/S24), ensure the primary 1x wide lens is used (macro auto-switching should be locked in camera settings).
2. **Lens Hygiene:**
   - Wipe the phone camera lens with a microfiber cloth before inserting into the cradle to prevent flare or hazy focus.
3. **Contact Hygiene:**
   - Wipe the skin-contact ring of the spacer with a 70% isopropyl alcohol wipe between each patient. Allow 15 seconds to air dry.
4. **Flash & Lighting:**
   - Keep phone LED flash disabled to prevent blinding specular reflections. Rely on ambient room lighting diffusing through the spacer side ports or integrated diffuse LED.

---

## 5. Local Codebase Setup

```bash
# Clone repository
git clone <repo-url> e-dermatologist && cd e-dermatologist

# Backend Setup
cd backend
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend Setup
cd ../frontend
npm install
```

---

## 6. Environment Variables (`.env`)

| Variable | Example | Description |
|---|---|---|
| `MODEL_PATH` | `models/model.onnx` | Path to trained EfficientNet-B0 ONNX weights |
| `CLASSES_PATH` | `configs/classes.yaml` | Class names and triage thresholds |
| `MAX_UPLOAD_MB` | `8` | Maximum image size limit |
| `ALLOWED_ORIGINS` | `*` | Allowed CORS origins (set to tunnel URL or `*` for testing) |
| `VITE_API_URL` | `https://api-tunnel.trycloudflare.com/api/v1` | URL for mobile PWA to reach backend |

---

## 7. Cloud Deployment (Backup for Hackathon Jury)

If presenting in an auditorium with unreliable local Wi-Fi, host the backend in a container on Google Cloud Run, Render, or Railway:

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Deploy the frontend static bundle to Vercel or Netlify with HTTPS enabled by default.

---

## 8. Mobile Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| **Camera permission prompt not appearing** | Connecting over plain `http://<ip>` | Must access via an HTTPS URL (Cloudflare Tunnel or ngrok). |
| **Image is black or obstructed** | Wrong rear lens selected on multi-camera phone | Tap camera switch icon or adjust phone position in spacer cradle. |
| **Blurry live preview** | Phone autofocus hunting or closer than minimum focal distance | Ensure 3D spacer height ($\ge 35\text{ mm}$) matches the phone camera minimum focal length. |
| **Severe colour cast** | Fluorescent or warm yellow bulb | Enable automatic colour constancy; position reference patch in frame. |
