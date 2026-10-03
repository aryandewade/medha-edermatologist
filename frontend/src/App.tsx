import React, { useState, useEffect } from 'react';
import { CameraViewfinder } from './components/CameraViewfinder';
import { ResultsDashboard } from './components/ResultsDashboard';
import { InstructionsModal } from './components/InstructionsModal';
import { checkHealth, predictImage, generateMockPrediction } from './api';
import { PredictResponse, Language } from './types';
import { translations } from './i18n';
import { Shield, Sparkles, HelpCircle, Activity, History } from 'lucide-react';

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
  const [showInstructions, setShowInstructions] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [currentLang, setCurrentLang] = useState<Language>('en');
  const [serverOnline, setServerOnline] = useState<boolean | null>(null);
  const [serverStatusText, setServerStatusText] = useState<string>('Checking backend...');
  const [sessionHistory, setSessionHistory] = useState<HistoryItem[]>([]);

  const t = translations[currentLang] || translations.en;

  useEffect(() => {
    handleTestConnection();
  }, []);

  const handleTestConnection = async () => {
    try {
      setServerStatusText('Pinging server...');
      const health = await checkHealth();
      setServerOnline(true);
      setServerStatusText(`Online (${health.num_classes} classes, ${health.active_modality})`);
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
    setSessionHistory(prev => [item, ...prev].slice(0, 8)); // keep last 8
  };

  const handleCapture = async (blob: Blob, previewUrl: string) => {
    setIsAnalyzing(true);
    setCapturedPreview(previewUrl);

    try {
      if (serverOnline) {
        const pred = await predictImage(blob, currentLang, true);
        setResult(pred);
        recordHistory(pred, previewUrl);
      } else {
        await new Promise(r => setTimeout(r, 1200));
        const mock = generateMockPrediction('eczema');
        setResult(mock);
        recordHistory(mock, previewUrl);
      }
    } catch (err: any) {
      console.error("Analysis error:", err);
      const mock = generateMockPrediction('eczema');
      mock.advice.message = `(Offline Demo Fallback) - Real API error: ${err.message}`;
      setResult(mock);
      recordHistory(mock, previewUrl);
    } finally {
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

  const handleLoadSample = (type: 'eczema' | 'suspicious' | 'healthy') => {
    setIsAnalyzing(true);
    const samplePreviews = {
      eczema: 'https://images.unsplash.com/photo-1588776814546-1ffcf47267a5?auto=format&fit=crop&w=400&q=80',
      suspicious: 'https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=400&q=80',
      healthy: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=400&q=80'
    };
    const preview = samplePreviews[type];
    setCapturedPreview(preview);

    setTimeout(() => {
      const mock = generateMockPrediction(type);
      setResult(mock);
      recordHistory(mock, preview);
      setIsAnalyzing(false);
    }, 700);
  };

  const handleSelectHistory = (item: HistoryItem) => {
    setResult(item.result);
    setCapturedPreview(item.previewUrl);
    setShowHistory(false);
  };

  const handleReset = () => {
    setResult(null);
    setCapturedPreview(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center">
      {/* 1. Header Bar */}
      <header className="w-full max-w-md px-4 py-3.5 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-40 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-500 flex items-center justify-center shadow-md shadow-blue-500/20">
            <Activity size={18} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
              {t.appTitle}
              <span className="text-[10px] font-normal uppercase px-1.5 py-0.2 bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded">
                {t.spacerBadge}
              </span>
            </h1>
            <p className="text-[10px] text-slate-400">{t.appSubtitle}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* History Button */}
          {sessionHistory.length > 0 && (
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="p-1.5 rounded-lg bg-slate-800/80 text-slate-300 hover:text-white border border-slate-700 transition-colors relative"
              title="Session History"
            >
              <History size={16} />
              <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-blue-600 text-[9px] font-bold flex items-center justify-center text-white">
                {sessionHistory.length}
              </span>
            </button>
          )}

          {/* Language Selector */}
          <div className="flex items-center bg-slate-800/80 border border-slate-700 rounded-lg p-0.5 text-[11px] font-medium">
            <button
              onClick={() => setCurrentLang('en')}
              className={`px-1.5 py-0.5 rounded ${currentLang === 'en' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}
            >
              EN
            </button>
            <button
              onClick={() => setCurrentLang('hi')}
              className={`px-1.5 py-0.5 rounded ${currentLang === 'hi' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}
            >
              HI
            </button>
            <button
              onClick={() => setCurrentLang('mr')}
              className={`px-1.5 py-0.5 rounded ${currentLang === 'mr' ? 'bg-blue-600 text-white' : 'text-slate-400'}`}
            >
              MR
            </button>
          </div>

          <button
            onClick={() => setShowInstructions(true)}
            className="p-1.5 rounded-lg bg-slate-800/80 text-slate-300 hover:text-white border border-slate-700 transition-colors"
            title="How to use spacer"
          >
            <HelpCircle size={16} />
          </button>
        </div>
      </header>

      {/* 2. Session History Drawer */}
      {showHistory && sessionHistory.length > 0 && (
        <div className="w-full max-w-md px-4 pt-3 pb-1 border-b border-slate-800 bg-slate-900/95 animate-fadeIn z-30">
          <div className="flex justify-between items-center text-xs font-semibold text-slate-300 mb-2">
            <span>Current Session Screenings ({sessionHistory.length})</span>
            <button
              onClick={() => setShowHistory(false)}
              className="text-slate-400 hover:text-white text-[11px]"
            >
              Close
            </button>
          </div>
          <div className="flex gap-2.5 overflow-x-auto pb-2 scrollbar-none">
            {sessionHistory.map((item) => (
              <button
                key={item.id}
                onClick={() => handleSelectHistory(item)}
                className="shrink-0 flex items-center gap-2 p-1.5 rounded-xl bg-slate-800/90 border border-slate-700 hover:border-blue-500 transition-all text-left"
              >
                <img
                  src={item.previewUrl}
                  alt="Thumb"
                  className="w-10 h-10 rounded-lg object-cover bg-slate-950"
                />
                <div className="pr-1 text-xs">
                  <span className="font-semibold text-white block max-w-[90px] truncate text-[11px]">
                    {t.conditions[item.result.prediction.class_id] || item.result.prediction.label}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">
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
        {/* Backend Status Link */}
        <div className="flex items-center justify-between px-2 text-[11px] text-slate-400">
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                serverOnline === true
                  ? 'bg-emerald-400 animate-pulse'
                  : serverOnline === false
                  ? 'bg-amber-400'
                  : 'bg-slate-500'
              }`}
            />
            <span>{serverStatusText}</span>
          </div>

          <button
            onClick={handleTestConnection}
            className="text-[11px] text-blue-400 hover:underline cursor-pointer"
          >
            Check Link
          </button>
        </div>

        {/* Viewfinder OR Results */}
        {!result ? (
          <div className="flex flex-col gap-4 animate-fadeIn">
            <CameraViewfinder
              onCapture={handleCapture}
              isAnalyzing={isAnalyzing}
              lang={currentLang}
              onUploadFallback={handleUploadFallback}
            />

            {/* Instant Demo Case Loaders */}
            <div className="glass-panel p-3.5 flex flex-col gap-2.5">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300 flex items-center gap-1">
                  <Sparkles size={13} className="text-cyan-400" />
                  {t.instantDemos}
                </span>
                <span className="text-[10px] text-slate-400">{t.demoHint}</span>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => handleLoadSample('eczema')}
                  disabled={isAnalyzing}
                  className="py-2 px-1 text-center text-xs font-medium rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-blue-300 transition-all active:scale-95"
                >
                  {t.conditions.eczema ? t.conditions.eczema.split(' ')[0] : 'Eczema'}
                </button>
                <button
                  onClick={() => handleLoadSample('suspicious')}
                  disabled={isAnalyzing}
                  className="py-2 px-1 text-center text-xs font-medium rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-red-900/50 text-red-400 transition-all active:scale-95"
                >
                  🚨 Suspicious
                </button>
                <button
                  onClick={() => handleLoadSample('healthy')}
                  disabled={isAnalyzing}
                  className="py-2 px-1 text-center text-xs font-medium rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-emerald-900/50 text-emerald-400 transition-all active:scale-95"
                >
                  {t.conditions.healthy ? t.conditions.healthy.split(' ')[0] : 'Healthy'}
                </button>
              </div>
            </div>

            {/* Physical Spacer Technology Info Box */}
            <div className="p-3.5 rounded-xl bg-slate-900/50 border border-slate-800/80 text-xs text-slate-400 leading-relaxed flex items-start gap-2.5">
              <Shield size={18} className="text-blue-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-slate-200 block font-medium">{t.howItWorksTitle}</strong>
                {t.howItWorksBody}
              </div>
            </div>
          </div>
        ) : (
          <ResultsDashboard
            result={result}
            previewUrl={capturedPreview}
            lang={currentLang}
            onReset={handleReset}
          />
        )}
      </main>

      {/* 4. Footer */}
      <footer className="w-full max-w-md px-4 py-3 text-center border-t border-slate-900 text-[10px] text-slate-400 flex flex-col gap-0.5">
        <div>{t.footerHospital}</div>
        <div>Screening decision aid • Not a definitive medical diagnosis</div>
      </footer>

      {/* Instructions Modal */}
      <InstructionsModal
        isOpen={showInstructions}
        onClose={() => setShowInstructions(false)}
      />
    </div>
  );
};
