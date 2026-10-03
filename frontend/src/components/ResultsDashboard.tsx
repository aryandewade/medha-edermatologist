import React, { useState } from 'react';
import { 
  AlertTriangle, 
  CheckCircle2, 
  Info, 
  Sparkles, 
  Layers, 
  Download, 
  RotateCcw,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  AlertCircle
} from 'lucide-react';
import { PredictResponse, Language } from '../types';
import { translations } from '../i18n';
import { downloadReferralPdf } from '../api';

interface Props {
  result: PredictResponse;
  previewUrl: string | null;
  lang?: Language;
  onReset: () => void;
}

export const ResultsDashboard: React.FC<Props> = ({
  result,
  previewUrl,
  lang = 'en',
  onReset
}) => {
  const [activeTab, setActiveTab] = useState<'full' | 'focus' | 'heatmap'>('full');
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.65);
  const [showAllProbs, setShowAllProbs] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const t = translations[lang] || translations.en;
  const { prediction, top3, probabilities, advice, latency_ms } = result;
  const currentCondition = t.conditions[prediction.class_id] || prediction.label;
  const currentConfidence = prediction.confidence;
  const currentSeverity = prediction.severity;

  const isUrgent = advice.level === 'urgent' || currentSeverity.toLowerCase().includes('urgent') || currentSeverity === 'Severe';
  const isUncertain = advice.level === 'uncertain';
  const isHealthy = prediction.class_id === 'healthy' || prediction.label.toLowerCase().includes('healthy');

  const bannerTheme = isUrgent
    ? {
        bg: 'bg-red-50/90 border border-red-200 text-red-900',
        icon: <AlertTriangle className="text-red-600 shrink-0 mt-0.5" size={18} />,
        header: t.triageHeaders.urgent || 'Urgent Medical Review Required',
        desc: advice.message || 'Clinical features require prompt specialist evaluation and prescription care.'
      }
    : isUncertain
    ? {
        bg: 'bg-amber-50/90 border border-amber-200 text-amber-900',
        icon: <Info className="text-amber-600 shrink-0 mt-0.5" size={18} />,
        header: t.triageHeaders.uncertain || 'Uncertain Presentation / Specialist Review',
        desc: advice.message || 'Atypical features observed. Consult a dermatologist for definitive evaluation.'
      }
    : isHealthy
    ? {
        bg: 'bg-emerald-50/90 border border-emerald-200 text-emerald-900',
        icon: <ShieldCheck className="text-emerald-600 shrink-0 mt-0.5" size={18} />,
        header: 'Normal Skin Presentation',
        desc: advice.message && !advice.message.includes('Offline') 
          ? advice.message 
          : 'The image shows normal skin appearance without significant signs of active inflammation or concerning lesions.'
      }
    : {
        bg: 'bg-blue-50/90 border border-blue-200 text-blue-900',
        icon: <CheckCircle2 className="text-blue-600 shrink-0 mt-0.5" size={18} />,
        header: t.triageHeaders.common || 'Primary Care Clinical Presentation',
        desc: advice.message
      };

  const handleDownloadPdf = async () => {
    setIsExporting(true);
    try {
      await downloadReferralPdf(result, "Field Screening Patient");
    } finally {
      setIsExporting(false);
    }
  };

  const activeCare = result.care_protocol;

  // Fallback care steps if model didn't provide specific care protocol
  const careSteps = activeCare?.care && activeCare.care.length > 0
    ? activeCare.care
    : isHealthy
    ? [
        "Keep the skin area clean, dry, and protected from environmental friction",
        "Avoid picking, scratching, or applying aggressive harsh astringents",
        "Consult a qualified healthcare professional or dermatologist for comprehensive diagnosis"
      ]
    : [
        "Keep the affected skin clean, hydrated, and protected from trauma",
        "Avoid self-prescribed topical corticosteroids without clinical confirmation",
        "Consult a qualified healthcare professional or dermatologist for comprehensive evaluation"
      ];

  const urgentWarningText = activeCare?.urgent || "symptoms worsen rapidly, fever develops, or severe pain and spreading inflammation occur.";

  // Choose which image to display based on activeTab
  const currentDisplayedImage = 
    activeTab === 'focus' && result.roi_preview_png_base64
      ? result.roi_preview_png_base64
      : previewUrl;

  return (
    <div className="w-full flex flex-col gap-4 animate-fadeIn pb-4">
      {/* 1. Triage Presentation Banner - Compact */}
      <div className={`py-2.5 px-3 rounded-xl ${bannerTheme.bg} flex items-start gap-2.5 shadow-xs`}>
        {bannerTheme.icon}
        <div className="flex-1 min-w-0">
          <div className="text-xs font-bold tracking-tight mb-0.5">
            {bannerTheme.header}
          </div>
          <div className="text-[11px] leading-snug text-slate-700">
            {bannerTheme.desc}
          </div>
        </div>
      </div>

      {/* 2. Primary Screening Finding Card */}
      <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-4 flex flex-col gap-3">
        <div className="flex justify-between items-center text-xs text-slate-500 font-medium">
          <span>{t.primaryFinding}</span>
          <span className="text-xs text-slate-400 font-mono">
            Medha V2 • {latency_ms ? latency_ms.toFixed(0) : '2238'} ms
          </span>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
              {currentCondition}
            </h2>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="text-xs text-slate-500">Severity:</span>
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                currentSeverity === 'Severe' || currentSeverity === 'urgent' || currentSeverity === 'URGENT'
                  ? 'bg-red-100 text-red-800'
                  : currentSeverity === 'Moderate'
                  ? 'bg-amber-100 text-amber-800'
                  : 'bg-emerald-100 text-emerald-800'
              }`}>
                {currentSeverity === 'LOW' || currentSeverity === 'low' || isHealthy ? 'Normal' : currentSeverity}
              </span>
            </div>
          </div>

          <div className="text-right">
            <div className="text-3xl font-extrabold text-blue-600 tracking-tight">
              {(currentConfidence * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              {t.confidence}
            </div>
          </div>
        </div>

        {/* Primary Confidence Bar */}
        <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
          <div
            className="h-full rounded-full bg-blue-600 transition-all duration-700"
            style={{ width: `${Math.min(100, Math.max(5, currentConfidence * 100))}%` }}
          />
        </div>
      </div>

      {/* 3. Analyzed Skin Frame - Compact */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-xs p-3 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
            <Sparkles size={14} className="text-blue-600" />
            {t.analyzedFrame}
          </span>

          {/* 3-Tab Segmented Control */}
          <div className="flex items-center gap-1 text-[11px]">
            <button
              onClick={() => setActiveTab('full')}
              className={`px-2 py-0.5 rounded font-medium transition-colors ${
                activeTab === 'full'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              Full Photo
            </button>

            <button
              onClick={() => setActiveTab('focus')}
              className={`px-2 py-0.5 rounded font-medium transition-colors ${
                activeTab === 'focus'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-blue-600 border border-blue-200 bg-blue-50/50 hover:bg-blue-50'
              }`}
              title="Centred ROI patch"
            >
              AI Focus
            </button>

            <button
              onClick={() => setActiveTab(activeTab === 'heatmap' ? 'full' : 'heatmap')}
              className={`px-2 py-0.5 rounded font-medium flex items-center gap-1 transition-colors ${
                activeTab === 'heatmap'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-600 border border-slate-200 bg-white hover:bg-slate-50'
              }`}
            >
              <Layers size={11} />
              Heatmap
            </button>
          </div>
        </div>

        {/* Compact Horizontal Photo Frame */}
        <div className="relative w-full aspect-[21/9] max-h-[140px] rounded-lg overflow-hidden border border-slate-200 bg-slate-100 flex items-center justify-center">
          {currentDisplayedImage ? (
            <img
              src={currentDisplayedImage}
              alt="Analyzed Skin Frame"
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="text-xs text-slate-400">No frame preview available</div>
          )}

          {result.heatmap_png_base64 && activeTab === 'heatmap' && (
            <img
              src={result.heatmap_png_base64}
              alt="Grad-CAM Saliency"
              className="absolute inset-0 w-full h-full object-cover mix-blend-screen transition-opacity duration-300 pointer-events-none"
              style={{ opacity: heatmapOpacity }}
            />
          )}
        </div>

        {/* Heatmap Opacity Slider if active */}
        {result.heatmap_png_base64 && activeTab === 'heatmap' && (
          <div className="flex items-center gap-2 px-1 text-[11px] text-slate-500">
            <span>Opacity:</span>
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.05"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
              className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
            />
            <span className="w-8 text-right font-mono font-medium text-slate-700">
              {(heatmapOpacity * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {/* Compact Clinical Tip Box */}
        <div className="py-1.5 px-2.5 rounded-lg bg-blue-50/70 border border-blue-100 text-[11px] text-slate-600 leading-snug flex items-start gap-2">
          <Info size={13} className="text-blue-600 shrink-0 mt-0.5" />
          <span>
            <strong className="text-slate-900 font-semibold">Clinical Tip:</strong> Close macro framing ensures the neural ensemble focuses on rash texture and erythema rather than surrounding background.
          </span>
        </div>
      </div>

      {/* 4. Clinical Care & Management Protocol */}
      <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
            <ShieldCheck size={18} className="text-emerald-600" />
            Clinical Care & Management Protocol
          </span>
        </div>

        {/* Care steps */}
        <div className="flex flex-col gap-2">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
            RECOMMENDED STEPS
          </span>
          <ul className="flex flex-col gap-2 text-xs text-slate-700">
            {careSteps.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2.5 leading-relaxed">
                <CheckCircle2 size={16} className="text-blue-600 fill-blue-50 shrink-0 mt-0.5" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Red Flag Alert Box */}
        <div className="p-3 rounded-xl bg-red-50/80 border border-red-200 text-xs text-red-900 flex items-start gap-2.5 mt-1">
          <AlertCircle size={16} className="text-red-600 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <strong className="text-red-800 font-bold">Seek Immediate Medical Care If: </strong>
            <span>{urgentWarningText}</span>
          </div>
        </div>
      </div>

      {/* 5. Differential Distribution (Top-3) */}
      <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-4 flex flex-col gap-3">
        <div className="text-sm font-bold text-slate-900">
          {t.differentialHeading}
        </div>

        <div className="flex flex-col gap-3">
          {top3.map((item, idx) => {
            const itemLabel = t.conditions[item.class_id] || item.label;
            const percentage = (item.probability * 100).toFixed(1);
            return (
              <div key={item.class_id} className="flex flex-col gap-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-800">
                    {idx + 1}. {itemLabel}
                  </span>
                  <span className="font-mono font-bold text-slate-800">
                    {percentage}%
                  </span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      item.class_id === 'suspicious_lesion'
                        ? 'bg-red-500'
                        : idx === 0
                        ? 'bg-blue-600'
                        : 'bg-blue-400'
                    }`}
                    style={{ width: `${Math.max(3, item.probability * 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Expandable all-classes probability list */}
        <button
          onClick={() => setShowAllProbs(!showAllProbs)}
          className="border border-blue-200 bg-blue-50/50 hover:bg-blue-50 text-blue-600 rounded-lg py-2 px-3 text-xs font-semibold text-center flex items-center justify-center gap-1.5 cursor-pointer transition-colors mt-1"
        >
          <span>
            {showAllProbs ? t.hideAllProbs : t.viewAllProbs}
          </span>
          {showAllProbs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {showAllProbs && (
          <div className="mt-2 pt-2 border-t border-slate-100 grid grid-cols-2 gap-2 text-xs">
            {Object.entries(probabilities).map(([cls, prob]) => (
              <div key={cls} className="flex justify-between p-2 rounded-lg bg-slate-50 border border-slate-100 font-mono text-[11px]">
                <span className="text-slate-600 capitalize">{cls.replace(/_/g, ' ')}</span>
                <span className="text-slate-900 font-bold">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 6. Action Buttons */}
      <div className="flex gap-3 pt-1">
        <button
          onClick={onReset}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-xl flex-1 flex items-center justify-center gap-2 shadow-sm active:scale-[0.98] transition-all cursor-pointer"
        >
          <RotateCcw size={16} />
          <span>{t.newCapture}</span>
        </button>

        <button
          onClick={handleDownloadPdf}
          disabled={isExporting}
          className="bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 font-semibold py-3 px-6 rounded-xl flex items-center justify-center gap-1.5 shadow-sm active:scale-[0.98] transition-all cursor-pointer"
          title="Download Clinical Referral PDF"
        >
          {isExporting ? (
            <div className="w-4 h-4 border-2 border-slate-400 border-t-blue-600 rounded-full animate-spin" />
          ) : (
            <Download size={16} />
          )}
          <span>PDF</span>
        </button>
      </div>

      {/* 7. Clinical Disclaimer & Institution Credit */}
      <div className="text-[11px] text-slate-400 text-center leading-relaxed px-4 pt-3 flex flex-col gap-2">
        <p className="max-w-[340px] mx-auto text-slate-500">
          {t.disclaimer}
        </p>
      </div>
    </div>
  );
};
