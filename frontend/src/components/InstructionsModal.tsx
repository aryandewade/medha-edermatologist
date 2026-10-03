import React from 'react';
import { X, Shield, HelpCircle } from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const InstructionsModal: React.FC<Props> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-fadeIn">
      <div className="w-full max-w-sm p-6 rounded-2xl bg-white border border-slate-200 shadow-2xl flex flex-col gap-4 text-slate-800">
        <div className="flex justify-between items-center border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <HelpCircle size={20} className="text-blue-600" />
            <h3 className="font-bold text-slate-900 text-base">Clinical Capture Protocol</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-full text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex flex-col gap-3.5 text-xs leading-relaxed text-slate-600">
          <div className="flex items-start gap-2.5">
            <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-[11px] shrink-0">
              1
            </span>
            <div>
              <strong className="text-slate-900 block font-semibold">Good Lighting</strong>
              Ensure the affected area is evenly illuminated without harsh shadows or specular glares.
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-[11px] shrink-0">
              2
            </span>
            <div>
              <strong className="text-slate-900 block font-semibold">Close Macro Framing</strong>
              Keep the camera 10–15 cm from the skin surface so the lesion fills the central target frame.
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-[11px] shrink-0">
              3
            </span>
            <div>
              <strong className="text-slate-900 block font-semibold">Clear Sharp Focus</strong>
              Hold steady until the skin texture and borders are sharp and blur-free.
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-[11px] shrink-0">
              4
            </span>
            <div>
              <strong className="text-slate-900 block font-semibold">Capture & Review</strong>
              Tap Capture & Analyze or Upload. The clinical triage engine will evaluate patterns and differential probabilities.
            </div>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-blue-50/80 border border-blue-100 text-[11px] text-slate-700 leading-normal flex items-start gap-2">
          <Shield size={16} className="text-blue-600 shrink-0 mt-0.5" />
          <span>
            <strong className="text-slate-900">Safety Notice:</strong> Rapidly expanding rash, high fever, or severe systemic distress requires urgent hospital emergency evaluation.
          </span>
        </div>

        <button
          onClick={onClose}
          className="btn-primary w-full py-2.5 text-xs font-semibold cursor-pointer"
        >
          Got It, Continue
        </button>
      </div>
    </div>
  );
};
