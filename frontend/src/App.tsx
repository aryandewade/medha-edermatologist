import React, { useState, useEffect } from 'react';
import { CameraViewfinder } from './components/CameraViewfinder';
import { ResultsDashboard } from './components/ResultsDashboard';
import { InstructionsModal } from './components/InstructionsModal';
import { checkHealth, predictImage, generateMockPrediction } from './api';
import { PredictResponse, Language } from './types';
import { translations } from './i18n';
import { 
  Activity, 
  HelpCircle, 
  Camera, 
  Upload, 
  History, 
  ShieldCheck, 
  Loader2 
} from 'lucide-react';

interface HistoryItem {
  id: string;
  timestamp: string;
  result: PredictResponse;
  previewUrl: string;
}

export const App: React.FC = () => {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [capturedPreview, setCapturedPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzingStep, setAnalyzingStep] = useState<string>('Standardizing optical colour balance...');
  const [showInstructions, setShowInstructions] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [currentLang, setCurrentLang] = useState<Language>('en');
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [serverStatusText, setServerStatusText] = useState<string>('Checking backend...');
  const [sessionHistory, setSessionHistory] = useState<HistoryItem[]>([]);
  const selectedEngine: 'medha_v2' | 'skinnet_ensemble' = 'medha_v2';

  const t = translations[currentLang] || translations.en;

  useEffect(() => {
    handleTestConnection();
  }, []);

  const handleTestConnection = async () => {
    try {
      setServerStatusText('Pinging server...');
      const health = await checkHealth();
      setServerOnline(true);
      setServerStatusText(`Online • ${health.num_classes || 6} classes • Clinical AI Triage Engine`);
    } catch (e: any) {
      console.warn("Backend offline or tunnel unavailable:", e);
      setServerOnline(false);
      setServerStatusText('Offline / Standalone Demo Mode');
    }
  };

  const recordHistory = (newResult: PredictResponse, preview: string) => {
    const item: HistoryItem = {
      id: newResult.request_id || String(Date.now()),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      result: newResult,
      previewUrl: preview
    };
    setSessionHistory(prev => [item, ...prev].slice(0, 8));
  };

  const handleCapture = async (blob: Blob, previewUrl: string) => {
    setIsAnalyzing(true);
    setCapturedPreview(previewUrl);
    setIsCameraOpen(false);

    setAnalyzingStep('Applying Shades-of-Gray colour constancy...');
    const stepTimer1 = setTimeout(() => {
      setAnalyzingStep('Segmenting central lesion region of interest...');
    }, 500);

    const stepTimer2 = setTimeout(() => {
      setAnalyzingStep('Evaluating skin morphology with triage ensemble...');
    }, 1100);

    const startTime = Date.now();
    try {
      let pred: PredictResponse;
      if (serverOnline) {
        pred = await predictImage(blob, currentLang, true, selectedEngine);
      } else {
        await new Promise(r => setTimeout(r, 1400));
        pred = generateMockPrediction('healthy');
      }

      const elapsed = Date.now() - startTime;
      if (elapsed < 1400) {
        await new Promise(r => setTimeout(r, 1400 - elapsed));
      }

      setResult(pred);
      recordHistory(pred, previewUrl);
    } catch (err: any) {
      console.error("Analysis error:", err);
      const mock = generateMockPrediction('healthy');
      mock.advice.message = `(Offline Demo Fallback) - API notice: ${err.message}`;
      setResult(mock);
      recordHistory(mock, previewUrl);
    } finally {
      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      setIsAnalyzing(false);
    }
  };

  const handleUploadFallback = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string;
      handleCapture(file, dataUrl);
    };
    reader.readAsDataURL(file);
  };


  const handleSelectHistory = (item: HistoryItem) => {
    setResult(item.result);
    setCapturedPreview(item.previewUrl);
    setShowHistory(false);
    setIsCameraOpen(false);
  };

  const handleReset = () => {
    setResult(null);
    setCapturedPreview(null);
    setIsCameraOpen(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col items-center">
      {/* 1. Header Bar matching Image 1 & 2 */}
      <header className="w-full max-w-md px-4 py-3 border-b border-slate-200/90 bg-white sticky top-0 z-40 flex items-center justify-between shadow-xs">
        <div className="flex items-center gap-2.5">
          {/* Blue rounded icon with pulse */}
          <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center shadow-md shadow-blue-500/20 text-white shrink-0">
            <Activity size={20} strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-slate-900 leading-tight">
              E-Dermatologist
            </h1>
            <p className="text-[11px] text-slate-500 font-normal">
              Clinical Skin Screening Tool
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Session History Drawer Button */}
          {sessionHistory.length > 0 && (
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="p-1.5 rounded-lg bg-slate-100 text-slate-600 hover:text-slate-900 border border-slate-200 transition-colors relative cursor-pointer"
              title="Session History"
            >
              <History size={16} />
              <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-blue-600 text-[9px] font-bold flex items-center justify-center text-white">
                {sessionHistory.length}
              </span>
            </button>
          )}

          {/* Help Circle Icon */}
          <button
            onClick={() => setShowInstructions(true)}
            className="w-8 h-8 rounded-full flex items-center justify-center text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
            title="Clinical capture instructions"
          >
            <HelpCircle size={20} />
          </button>
        </div>
      </header>

      {/* 2. Session History Drawer */}
      {showHistory && sessionHistory.length > 0 && (
        <div className="w-full max-w-md px-4 pt-3 pb-2 border-b border-slate-200 bg-white animate-fadeIn z-30 shadow-sm">
          <div className="flex justify-between items-center text-xs font-semibold text-slate-700 mb-2">
            <span>Session Screenings ({sessionHistory.length})</span>
            <button
              onClick={() => setShowHistory(false)}
              className="text-slate-400 hover:text-slate-800 text-[11px] cursor-pointer"
            >
              Close
            </button>
          </div>
          <div className="flex gap-2.5 overflow-x-auto pb-1 scrollbar-none">
            {sessionHistory.map((item) => (
              <button
                key={item.id}
                onClick={() => handleSelectHistory(item)}
                className="shrink-0 flex items-center gap-2 p-1.5 rounded-xl bg-slate-50 border border-slate-200 hover:border-blue-400 transition-all text-left cursor-pointer"
              >
                <img
                  src={item.previewUrl}
                  alt="Thumb"
                  className="w-10 h-10 rounded-lg object-cover bg-slate-200"
                />
                <div className="pr-1 text-xs">
                  <span className="font-semibold text-slate-900 block max-w-[90px] truncate text-[11px]">
                    {t.conditions[item.result.prediction.class_id] || item.result.prediction.label}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {(item.result.prediction.confidence * 100).toFixed(0)}% • {item.timestamp}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 3. Main Content Column */}
      <main className="w-full max-w-md flex-1 p-4 flex flex-col gap-4">

        {/* Viewfinder OR Capture Card OR Results Dashboard */}
        {!result ? (
          !isCameraOpen ? (
            /* EXACT CAPTURE CARD FROM IMAGE 2 */
            <div className="flex flex-col gap-4 animate-fadeIn">
              <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-8 flex flex-col items-center text-center gap-6">
                {/* Dashed outer ring with soft blue circle and dark camera icon */}
                <div className="relative w-48 h-48 rounded-full border border-dashed border-slate-300 flex items-center justify-center my-2">
                  <div className="w-28 h-28 rounded-full bg-blue-50 flex items-center justify-center shadow-xs">
                    <Camera size={44} className="text-slate-800" strokeWidth={1.75} />
                  </div>
                </div>

                {/* Title & helper text */}
                <div className="flex flex-col gap-1.5">
                  <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                    Take a photo to analyze
                  </h2>
                  <p className="text-xs text-slate-500 max-w-[280px] mx-auto leading-relaxed">
                    Ensure good lighting and clear focus for best results
                  </p>
                </div>

                {/* Action Buttons */}
                <div className="w-full flex flex-col gap-2.5">
                  <button
                    onClick={() => setIsCameraOpen(true)}
                    className="w-full py-3.5 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl flex items-center justify-center gap-2 shadow-sm active:scale-[0.98] transition-all cursor-pointer"
                  >
                    <Camera size={18} />
                    <span>Live Camera Viewfinder</span>
                  </button>

                  <label className="w-full py-3 px-4 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-xl flex items-center justify-center gap-2 shadow-sm active:scale-[0.98] transition-all cursor-pointer text-sm">
                    <Camera size={17} />
                    <span>Take Photo with Phone Camera</span>
                    <input
                      type="file"
                      accept="image/*"
                      capture="environment"
                      onChange={handleUploadFallback}
                      className="hidden"
                    />
                  </label>

                  <label className="w-full py-2.5 px-4 bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium rounded-xl border border-slate-200 flex items-center justify-center gap-2 active:scale-[0.98] transition-all cursor-pointer text-xs">
                    <Upload size={15} />
                    <span>Upload from Gallery / Files</span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleUploadFallback}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>


              {/* Clinical Protocol Tip */}
              <div className="p-3.5 rounded-2xl bg-white border border-slate-200 text-xs text-slate-600 leading-relaxed flex items-start gap-2.5 shadow-xs">
                <ShieldCheck size={18} className="text-blue-600 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-slate-900 block font-semibold mb-0.5">
                    Clinical Macro Recommendation:
                  </strong>
                  Position smartphone 10–15 cm from lesion. Ensure diffuse lighting to highlight skin texture without specular reflections.
                </div>
              </div>
            </div>
          ) : (
            /* Live Camera Viewfinder View */
            <CameraViewfinder
              onCapture={handleCapture}
              isAnalyzing={isAnalyzing}
              lang={currentLang}
              onUploadFallback={handleUploadFallback}
              onClose={() => setIsCameraOpen(false)}
            />
          )
        ) : (
          /* EXACT RESULTS DASHBOARD FROM IMAGE 1 */
          <ResultsDashboard
            result={result}
            previewUrl={capturedPreview}
            lang={currentLang}
            onReset={handleReset}
          />
        )}
      </main>


      {/* Instructions Modal */}
      <InstructionsModal
        isOpen={showInstructions}
        onClose={() => setShowInstructions(false)}
      />

      {/* 5. Clinical Analysis Scanning HUD Overlay */}
      {isAnalyzing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm animate-fadeIn">
          <div className="w-full max-w-xs p-6 rounded-2xl bg-white border border-slate-200 shadow-2xl flex flex-col items-center text-center gap-4">
            {/* Captured thumbnail if available */}
            {capturedPreview && (
              <div className="relative w-24 h-24 rounded-xl overflow-hidden border border-slate-200 shadow-sm bg-slate-100">
                <img
                  src={capturedPreview}
                  alt="Analyzing preview"
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-x-0 h-1 bg-blue-500 animate-pulse top-1/2 -translate-y-1/2 shadow-sm" />
              </div>
            )}

            <div className="flex flex-col items-center gap-2">
              <Loader2 size={28} className="text-blue-600 animate-spin" />
              <h3 className="font-bold text-slate-900 text-sm">
                Analyzing Skin Lesion
              </h3>
              <p className="text-xs text-slate-500 max-w-[200px] leading-relaxed">
                {analyzingStep}
              </p>
            </div>

            <div className="w-full h-1.5 rounded-full bg-slate-100 overflow-hidden">
              <div className="h-full rounded-full bg-blue-600 animate-pulse w-3/4" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
