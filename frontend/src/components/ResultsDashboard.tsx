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
  ChevronUp
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
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.65);
  const [showAllProbs, setShowAllProbs] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const t = translations[lang] || translations.en;
  const { prediction, top3, probabilities, advice, quality, latency_ms } = result;

  const isUrgent = advice.level === 'urgent' || prediction.severity === 'urgent';
  const isUncertain = advice.level === 'uncertain';
  const isHealthy = prediction.class_id === 'healthy';

  const bannerTheme = isUrgent
    ? {
        bg: 'bg-red-950/60 border-red-500/50 text-red-200',
        icon: <AlertTriangle className="text-red-400 shrink-0" size={24} />,
        header: t.triageHeaders.urgent
      }
    : isUncertain
    ? {
        bg: 'bg-amber-950/60 border-amber-500/50 text-amber-200',
        icon: <Info className="text-amber-400 shrink-0" size={24} />,
        header: t.triageHeaders.uncertain
      }
    : isHealthy
    ? {
        bg: 'bg-emerald-950/60 border-emerald-500/50 text-emerald-200',
        icon: <ShieldCheck className="text-emerald-400 shrink-0" size={24} />,
        header: t.triageHeaders.normal
      }
    : {
        bg: 'bg-blue-950/60 border-blue-500/50 text-blue-200',
        icon: <CheckCircle2 className="text-blue-400 shrink-0" size={24} />,
        header: t.triageHeaders.common
      };

  const handleDownloadPdf = async () => {
    setIsExporting(true);
    try {
      await downloadReferralPdf(result, "Field Screening Patient");
    } finally {
      setIsExporting(false);
    }
  };

  const translatedLabel = t.conditions[prediction.class_id] || prediction.label;

  return (
    <div className="w-full flex flex-col gap-4 animate-fadeIn pb-6">
      {/* 1. Triage Referral Banner */}
      <div className={`p-4 rounded-2xl border backdrop-blur-md flex items-start gap-3 shadow-lg ${bannerTheme.bg}`}>
        {bannerTheme.icon}
        <div className="flex-1">
          <div className="text-xs font-bold uppercase tracking-wider mb-1 opacity-90">
            {bannerTheme.header}
          </div>
          <div className="text-sm font-semibold text-white mb-1">
            {advice.title}
          </div>
          <div className="text-xs opacity-90 leading-relaxed">
            {advice.message}
          </div>
        </div>
      </div>

      {/* 2. Top Finding Card */}
      <div className="glass-panel p-4 flex flex-col gap-3">
        <div className="flex justify-between items-center text-xs text-slate-400 font-medium">
          <span>{t.primaryFinding}</span>
          <span className="text-[11px] text-slate-400 font-mono">Latency: {latency_ms.toFixed(0)} ms</span>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">
              {translatedLabel}
            </h2>
            <p className="text-xs text-slate-400 capitalize mt-0.5">
              Severity: <span className="font-medium text-slate-200">{prediction.severity}</span>
            </p>
          </div>

          <div className="text-right">
            <div className="text-3xl font-extrabold text-blue-400 tracking-tight">
              {(prediction.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-[11px] text-slate-400 uppercase tracking-wide">
              {t.confidence}
            </div>
          </div>
        </div>

        {/* Primary Confidence Bar */}
        <div className="prob-bar-container">
          <div
            className="prob-bar-fill bg-gradient-to-r from-blue-600 to-cyan-400"
            style={{ width: `${Math.min(100, prediction.confidence * 100)}%` }}
          />
        </div>
      </div>

      {/* 3. Image & Grad-CAM Heatmap Toggle */}
      <div className="glass-panel p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
            <Sparkles size={14} className="text-blue-400" />
            {t.analyzedFrame}
          </span>

          {result.heatmap_png_base64 && (
            <button
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`text-xs px-2.5 py-1 rounded-full border transition-all flex items-center gap-1 ${
                showHeatmap
                  ? 'bg-blue-600 text-white border-blue-500'
                  : 'bg-slate-800 text-slate-300 border-slate-700'
              }`}
            >
              <Layers size={12} />
              {showHeatmap ? t.heatmapOn : t.heatmapOff}
            </button>
          )}
        </div>

        <div className="relative aspect-square w-full max-w-[280px] mx-auto rounded-xl overflow-hidden border border-slate-800 bg-slate-950 shadow-inner">
          {previewUrl && (
            <img
              src={previewUrl}
              alt="Analyzed Skin"
              className="w-full h-full object-cover"
            />
          )}

          {result.heatmap_png_base64 && showHeatmap && (
            <img
              src={result.heatmap_png_base64}
              alt="Grad-CAM Saliency"
              className="absolute inset-0 w-full h-full object-cover mix-blend-screen transition-opacity duration-300"
              style={{ opacity: heatmapOpacity }}
            />
          )}
        </div>

        {result.heatmap_png_base64 && showHeatmap && (
          <div className="flex items-center gap-3 px-2 pt-1 text-xs text-slate-400">
            <span>Opacity:</span>
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.05"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
              className="w-full accent-blue-500 cursor-pointer h-1.5 bg-slate-700 rounded-lg"
            />
            <span className="w-8 text-right font-mono">{(heatmapOpacity * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>

      {/* 4. Top-3 Differential Distribution */}
      <div className="glass-panel p-4 flex flex-col gap-3">
        <div className="text-xs font-semibold text-slate-300">
          {t.differentialHeading}
        </div>

        <div className="flex flex-col gap-2.5">
          {top3.map((item, idx) => {
            const itemLabel = t.conditions[item.class_id] || item.label;
            return (
              <div key={item.class_id} className="flex flex-col gap-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-medium text-slate-200">
                    {idx + 1}. {itemLabel}
                  </span>
                  <span className="font-mono text-slate-300">
                    {(item.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="prob-bar-container">
                  <div
                    className={`prob-bar-fill ${
                      item.class_id === 'suspicious_lesion'
                        ? 'bg-red-500'
                        : idx === 0
                        ? 'bg-blue-500'
                        : 'bg-slate-500'
                    }`}
                    style={{ width: `${Math.max(2, item.probability * 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Expandable all-classes probability list */}
        <button
          onClick={() => setShowAllProbs(!showAllProbs)}
          className="text-[11px] text-blue-400 hover:text-blue-300 transition-colors flex items-center justify-center gap-1 pt-1"
        >
          {showAllProbs ? (
            <>{t.hideAllProbs} <ChevronUp size={12} /></>
          ) : (
            <>{t.viewAllProbs} <ChevronDown size={12} /></>
          )}
        </button>

        {showAllProbs && (
          <div className="mt-2 pt-2 border-t border-slate-800 grid grid-cols-2 gap-2 text-xs">
            {Object.entries(probabilities).map(([cls, prob]) => (
              <div key={cls} className="flex justify-between p-1.5 rounded bg-slate-900/60 font-mono text-[11px]">
                <span className="text-slate-400 capitalize">{cls.replace('_', ' ')}</span>
                <span className="text-slate-200 font-semibold">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 5. Quality & Calibration Details */}
      <div className="glass-panel p-4 flex flex-col gap-2 text-xs text-slate-400">
        <div className="font-semibold text-slate-300 mb-1 flex items-center justify-between">
          <span>{t.opticalMetrics}</span>
          <span className="status-chip success text-[10px] py-0.5 px-2">
            {t.statusGood}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
          <div className="bg-slate-900/50 p-2 rounded border border-slate-800/80">
            <span className="text-slate-500 block text-[10px] uppercase">Sharpness (Laplacian)</span>
            <span className="text-slate-200 font-bold">{quality.blur_score.toFixed(1)}</span>
            <span className="text-[9px] text-slate-500 block">Min: 55.0</span>
          </div>

          <div className="bg-slate-900/50 p-2 rounded border border-slate-800/80">
            <span className="text-slate-500 block text-[10px] uppercase">Mean Brightness</span>
            <span className="text-slate-200 font-bold">{quality.brightness.toFixed(1)}</span>
            <span className="text-[9px] text-slate-500 block">Range: [40, 225]</span>
          </div>

          <div className="bg-slate-900/50 p-2 rounded border border-slate-800/80">
            <span className="text-slate-500 block text-[10px] uppercase">Specular Glare</span>
            <span className="text-slate-200 font-bold">{(quality.glare_ratio * 100).toFixed(1)}%</span>
            <span className="text-[9px] text-slate-500 block">Max: 12%</span>
          </div>

          <div className="bg-slate-900/50 p-2 rounded border border-slate-800/80">
            <span className="text-slate-500 block text-[10px] uppercase">Colour Constancy</span>
            <span className="text-emerald-400 font-bold">Shades-of-Gray</span>
            <span className="text-[9px] text-slate-500 block">p=6 Minkowski</span>
          </div>
        </div>
      </div>

      {/* 6. Action Buttons */}
      <div className="flex gap-3">
        <button
          onClick={onReset}
          className="btn-primary flex-1 shadow-lg"
        >
          <RotateCcw size={18} />
          <span>{t.newCapture}</span>
        </button>

        <button
          onClick={handleDownloadPdf}
          disabled={isExporting}
          className="btn-secondary px-4 text-xs font-semibold flex items-center gap-1.5"
          title="Download Referral PDF Summary"
        >
          {isExporting ? (
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <Download size={16} />
          )}
          <span>PDF</span>
        </button>
      </div>

      {/* Medical Disclaimer */}
      <div className="text-[11px] text-slate-500 text-center leading-relaxed px-3 pt-2">
        {t.disclaimer}
      </div>
    </div>
  );
};
