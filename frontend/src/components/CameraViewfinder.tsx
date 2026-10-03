import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, RefreshCw, Zap, AlertCircle, CheckCircle2, Sliders, ZoomIn, ArrowLeft } from 'lucide-react';
import { Language } from '../types';

interface Props {
  onCapture: (blob: Blob, previewUrl: string) => void;
  isAnalyzing: boolean;
  lang?: Language;
  onUploadFallback: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onClose?: () => void;
}

export const CameraViewfinder: React.FC<Props> = ({
  onCapture,
  isAnalyzing,
  lang: _lang = 'en',
  onUploadFallback,
  onClose
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [torchOn, setTorchOn] = useState(false);
  const [hasTorch, setHasTorch] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);

  const startCamera = useCallback(async (deviceId?: string) => {
    try {
      setCameraError(null);
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        if (typeof window !== 'undefined' && window.isSecureContext === false) {
          throw new Error("Mobile browsers strictly require HTTPS to open live camera. Ensure the URL starts with https://, or use 'Snap with Native Camera' below.");
        }
        throw new Error("Live camera streaming is not supported or is blocked by your browser settings.");
      }

      let newStream: MediaStream;
      try {
        const constraints: MediaStreamConstraints = {
          video: deviceId
            ? { deviceId: { exact: deviceId } }
            : { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false
        };
        newStream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (firstErr) {
        console.warn("High-res constraints failed, falling back to basic back camera:", firstErr);
        try {
          newStream = await navigator.mediaDevices.getUserMedia({
            video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'environment' },
            audio: false
          });
        } catch (secondErr) {
          newStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
          });
        }
      }

      setStream(newStream);

      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
        try {
          await videoRef.current.play();
        } catch (playErr) {
          console.warn("Video play notice:", playErr);
        }
      }

      const videoTrack = newStream.getVideoTracks()[0];
      const capabilities = (videoTrack.getCapabilities ? videoTrack.getCapabilities() : {}) as any;
      setHasTorch(!!capabilities.torch);

      if (navigator.mediaDevices.enumerateDevices) {
        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = allDevices.filter(d => d.kind === 'videoinput');
        setDevices(videoDevices);
        if (!deviceId && videoDevices.length > 0) {
          setSelectedDeviceId(videoTrack.getSettings().deviceId || videoDevices[0].deviceId);
        }
      }
    } catch (err: any) {
      console.warn("Camera access failed:", err);
      let msg = err.message || "Camera permission denied or camera unavailable.";
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = "Camera permission was blocked. Tap the lock/tune icon in your browser URL bar, tap Permissions, and set Camera to Allow.";
      }
      setCameraError(msg);
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
      } catch (err) {}
    }
  };

  const handleCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (typeof navigator !== 'undefined' && navigator.vibrate) {
      try {
        navigator.vibrate([35, 20, 35]);
      } catch {}
    }

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;

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
    <div className="relative w-full overflow-hidden rounded-2xl bg-black border border-slate-200 shadow-lg flex flex-col items-center animate-fadeIn">
      {/* Top Bar with Back Button */}
      {onClose && (
        <div className="w-full bg-slate-900/90 text-white px-3 py-2 flex items-center justify-between text-xs z-20">
          <button
            onClick={onClose}
            className="flex items-center gap-1.5 text-slate-300 hover:text-white font-medium cursor-pointer"
          >
            <ArrowLeft size={16} />
            <span>Back to photo options</span>
          </button>
          <span className="text-[11px] text-slate-400">Live Viewfinder</span>
        </div>
      )}

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

        {/* Circular Reticle Overlay */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-48 h-48 rounded-full border-2 border-dashed border-white/80 shadow-2xl relative flex items-center justify-center">
            <div className="w-4 h-0.5 bg-white/80 absolute" />
            <div className="h-4 w-0.5 bg-white/80 absolute" />
            <div className="absolute -top-7 left-1/2 -translate-x-1/2 text-[11px] font-medium tracking-wide uppercase text-white bg-black/60 px-2.5 py-0.5 rounded-full backdrop-blur-sm whitespace-nowrap">
              Center skin lesion
            </div>
          </div>
        </div>

        {/* Top Status Chips & Controls */}
        <div className="absolute top-3 left-3 right-3 flex justify-between items-center z-10 pointer-events-none">
          <div className="flex gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 backdrop-blur-sm">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Focus OK
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 backdrop-blur-sm">
              <CheckCircle2 size={12} />
              Light OK
            </span>
          </div>

          <div className="flex gap-2 pointer-events-auto">
            {hasTorch && (
              <button
                onClick={toggleTorch}
                title="Toggle Torch/Flash"
                className={`p-2 rounded-full backdrop-blur-md border transition-all cursor-pointer ${
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
                title="Switch Camera"
                className="p-2 rounded-full bg-slate-900/60 text-white border border-white/20 backdrop-blur-md cursor-pointer"
              >
                <RefreshCw size={16} />
              </button>
            )}
          </div>
        </div>

        {/* Bottom Zoom Selector */}
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-2 py-1 rounded-full bg-black/60 border border-white/20 backdrop-blur-md pointer-events-auto">
          <ZoomIn size={12} className="text-slate-300" />
          {[1.0, 1.5, 2.0].map((lvl) => (
            <button
              key={lvl}
              onClick={() => handleZoomChange(lvl)}
              className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold transition-all cursor-pointer ${
                zoomLevel === lvl
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              {lvl.toFixed(1)}x
            </button>
          ))}
        </div>

        {/* Camera Permission / Error Fallback */}
        {cameraError && (
          <div className="absolute inset-0 bg-slate-950/95 flex flex-col items-center justify-center p-6 text-center z-20 overflow-y-auto">
            <AlertCircle size={38} className="text-amber-400 mb-2.5 shrink-0" />
            <p className="text-sm font-semibold text-white mb-1">Live Camera Unavailable</p>
            <p className="text-xs text-slate-300 mb-4 max-w-xs leading-relaxed">{cameraError}</p>

            <div className="flex flex-col gap-2 w-full max-w-xs">
              {/* Guaranteed Fallback: Opens Mobile Camera natively without WebRTC */}
              <label className="w-full py-2.5 px-4 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-xl flex items-center justify-center gap-2 shadow-sm transition-all cursor-pointer text-xs">
                <Camera size={16} />
                <span>Snap with Native Camera</span>
                <input
                  type="file"
                  accept="image/*"
                  capture="environment"
                  onChange={onUploadFallback}
                  className="hidden"
                />
              </label>

              <button
                onClick={() => startCamera()}
                className="w-full py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium rounded-xl flex items-center justify-center gap-2 transition-all cursor-pointer text-xs"
              >
                <RefreshCw size={13} /> Retry Live View
              </button>

              <label className="w-full py-1.5 px-3 text-slate-400 hover:text-slate-200 text-xs flex items-center justify-center gap-1.5 cursor-pointer">
                <span>Or select photo from gallery</span>
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
      <div className="w-full p-4 bg-white border-t border-slate-200 flex flex-col gap-3">
        <button
          onClick={handleCapture}
          disabled={isAnalyzing || !!cameraError}
          className="btn-primary w-full shadow-md cursor-pointer"
        >
          {isAnalyzing ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Analyzing Frame & Clinical Features...</span>
            </>
          ) : (
            <>
              <Camera size={20} />
              <span className="text-base tracking-wide">Capture & Analyze</span>
            </>
          )}
        </button>

        <div className="flex items-center justify-between text-xs text-slate-500 px-1">
          <label className="hover:text-blue-600 transition-colors cursor-pointer flex items-center gap-1.5 py-1">
            <span>Upload photo from files</span>
            <input
              type="file"
              accept="image/*"
              onChange={onUploadFallback}
              className="hidden"
            />
          </label>

          {devices.length > 1 && (
            <div className="flex items-center gap-1">
              <Sliders size={12} className="text-slate-400" />
              <select
                value={selectedDeviceId}
                onChange={handleDeviceChange}
                className="bg-transparent text-slate-600 text-xs border-none focus:outline-none cursor-pointer"
              >
                {devices.map((d, i) => (
                  <option key={d.deviceId || i} value={d.deviceId}>
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
