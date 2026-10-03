import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, RefreshCw, Zap, AlertCircle, CheckCircle2, Sliders, ZoomIn } from 'lucide-react';
import { Language } from '../types';
import { translations } from '../i18n';

interface Props {
  onCapture: (blob: Blob, previewUrl: string) => void;
  isAnalyzing: boolean;
  lang?: Language;
  onUploadFallback: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export const CameraViewfinder: React.FC<Props> = ({
  onCapture,
  isAnalyzing,
  lang = 'en',
  onUploadFallback
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const t = translations[lang] || translations.en;

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [torchOn, setTorchOn] = useState(false);
  const [hasTorch, setHasTorch] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [zoomLevel, setZoomLevel] = useState<number>(1.0); // 1.0x, 1.5x, 2.0x

  const startCamera = useCallback(async (deviceId?: string) => {
    try {
      setCameraError(null);
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId } }
          : { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false
      };

      const newStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(newStream);

      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
      }

      const videoTrack = newStream.getVideoTracks()[0];
      const capabilities = (videoTrack.getCapabilities ? videoTrack.getCapabilities() : {}) as any;
      setHasTorch(!!capabilities.torch);

      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter(d => d.kind === 'videoinput');
      setDevices(videoDevices);
      if (!deviceId && videoDevices.length > 0) {
        setSelectedDeviceId(videoTrack.getSettings().deviceId || videoDevices[0].deviceId);
      }
    } catch (err: any) {
      console.warn("Camera access failed:", err);
      setCameraError(err.message || "Camera permission denied or camera unavailable.");
    }
  }, [stream]);

  useEffect(() => {
    startCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  const toggleTorch = async () => {
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    try {
      await (track as any).applyConstraints({
        advanced: [{ torch: !torchOn }]
      });
      setTorchOn(!torchOn);
    } catch (e) {
      console.warn("Torch toggle not supported:", e);
    }
  };

  const handleDeviceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    setSelectedDeviceId(id);
    startCamera(id);
  };

  const handleZoomChange = (level: number) => {
    setZoomLevel(level);
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    const capabilities = (track.getCapabilities ? track.getCapabilities() : {}) as any;
    if (capabilities.zoom) {
      const min = capabilities.zoom.min || 1;
      const max = capabilities.zoom.max || 3;
      const targetZoom = Math.min(max, Math.max(min, level));
      try {
        (track as any).applyConstraints({ advanced: [{ zoom: targetZoom }] });
      } catch (err) {
        // Fallback to CSS digital zoom handled below
      }
    }
  };

  const handleCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Haptic feedback
    if (typeof navigator !== 'undefined' && navigator.vibrate) {
      try {
        navigator.vibrate([35, 20, 35]);
      } catch {}
    }

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;

    // Apply digital zoom crop if zoomLevel > 1.0
    const cropW = vw / zoomLevel;
    const cropH = vh / zoomLevel;
    const cropX = (vw - cropW) / 2;
    const cropY = (vh - cropH) / 2;

    canvas.width = cropW;
    canvas.height = cropH;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    
    canvas.toBlob((blob) => {
      if (blob) {
        const previewUrl = canvas.toDataURL('image/jpeg', 0.92);
        onCapture(blob, previewUrl);
      }
    }, 'image/jpeg', 0.92);
  };

  return (
    <div className="relative w-full overflow-hidden rounded-2xl bg-black border border-slate-800 shadow-2xl flex flex-col items-center">
      {/* Viewport Frame */}
      <div className="relative w-full aspect-[4/3] max-h-[460px] overflow-hidden flex items-center justify-center bg-slate-950">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'center center' }}
          className="w-full h-full object-cover transition-transform duration-200"
        />

        {/* Circular 3D Spacer Reticle Overlay */}
        <div className="spacer-reticle-container">
          <div className={`spacer-ring ${isAnalyzing ? 'scanning' : ''}`}>
            <div className="spacer-crosshair" />
            
            <div className="absolute -top-7 left-1/2 -translate-x-1/2 text-[11px] font-medium tracking-wide uppercase text-slate-300/80 bg-slate-900/70 px-2.5 py-0.5 rounded-full backdrop-blur-sm whitespace-nowrap">
              {t.alignSpacer}
            </div>
          </div>
        </div>

        {/* Top Status Chips */}
        <div className="absolute top-3 left-3 right-3 flex justify-between items-center z-10 pointer-events-none">
          <div className="flex gap-2">
            <span className="status-chip success">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              {t.focusOk}
            </span>
            <span className="status-chip success">
              <CheckCircle2 size={12} />
              {t.lightOk}
            </span>
          </div>

          <div className="flex gap-2 pointer-events-auto">
            {hasTorch && (
              <button
                onClick={toggleTorch}
                title="Toggle Torch/Flash"
                className={`p-2 rounded-full backdrop-blur-md border transition-all ${
                  torchOn ? 'bg-amber-400 text-slate-950 border-amber-300' : 'bg-slate-900/60 text-white border-white/20'
                }`}
              >
                <Zap size={16} />
              </button>
            )}

            {devices.length > 1 && (
              <button
                onClick={() => {
                  const idx = devices.findIndex(d => d.deviceId === selectedDeviceId);
                  const nextIdx = (idx + 1) % devices.length;
                  setSelectedDeviceId(devices[nextIdx].deviceId);
                  startCamera(devices[nextIdx].deviceId);
                }}
                title="Switch Lens"
                className="p-2 rounded-full bg-slate-900/60 text-white border border-white/20 backdrop-blur-md"
              >
                <RefreshCw size={16} />
              </button>
            )}
          </div>
        </div>

        {/* Bottom Floating Macro Zoom Selector */}
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-2 py-1 rounded-full bg-slate-950/75 border border-white/10 backdrop-blur-md pointer-events-auto">
          <ZoomIn size={12} className="text-slate-400" />
          {[1.0, 1.5, 2.0].map((lvl) => (
            <button
              key={lvl}
              onClick={() => handleZoomChange(lvl)}
              className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold transition-all ${
                zoomLevel === lvl
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {lvl.toFixed(1)}x
            </button>
          ))}
        </div>

        {/* Camera Permission / Error Fallback */}
        {cameraError && (
          <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-6 text-center z-20">
            <AlertCircle size={40} className="text-amber-400 mb-3" />
            <p className="text-sm font-semibold text-slate-200 mb-1">Camera Inactive</p>
            <p className="text-xs text-slate-400 mb-4 max-w-xs">{cameraError}</p>
            <div className="flex flex-col gap-2 w-full max-w-xs">
              <button
                onClick={() => startCamera()}
                className="btn-secondary text-xs w-full py-2.5"
              >
                <RefreshCw size={14} /> Retry Camera
              </button>
              <label className="btn-primary text-xs w-full py-2.5 cursor-pointer">
                <span>{t.uploadPhoto}</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={onUploadFallback}
                  className="hidden"
                />
              </label>
            </div>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {/* Primary Capture Action Bar */}
      <div className="w-full p-4 bg-slate-950/95 border-t border-slate-800 flex flex-col gap-3">
        <button
          onClick={handleCapture}
          disabled={isAnalyzing || !!cameraError}
          className="btn-primary w-full shadow-lg shadow-blue-600/30"
        >
          {isAnalyzing ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>{t.analyzingBtn}</span>
            </>
          ) : (
            <>
              <Camera size={20} />
              <span className="text-base tracking-wide">{t.captureBtn}</span>
            </>
          )}
        </button>

        <div className="flex items-center justify-between text-xs text-slate-400 px-1">
          <label className="hover:text-blue-400 transition-colors cursor-pointer flex items-center gap-1.5 py-1">
            <span>{t.uploadPhoto}</span>
            <input
              type="file"
              accept="image/*"
              onChange={onUploadFallback}
              className="hidden"
            />
          </label>

          {devices.length > 1 && (
            <div className="flex items-center gap-1">
              <Sliders size={12} className="text-slate-500" />
              <select
                value={selectedDeviceId}
                onChange={handleDeviceChange}
                className="bg-transparent text-slate-300 text-xs border-none focus:outline-none cursor-pointer"
              >
                {devices.map((d, i) => (
                  <option key={d.deviceId || i} value={d.deviceId} className="bg-slate-900 text-white">
                    {d.label || `Camera ${i + 1}`}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
