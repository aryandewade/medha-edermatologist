import React from 'react';
import { X, Shield, HelpCircle } from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const InstructionsModal: React.FC<Props> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fadeIn">
      <div className="glass-panel w-full max-w-sm p-5 border border-slate-700 bg-slate-900/95 shadow-2xl flex flex-col gap-4 text-slate-200">
        <div className="flex justify-between items-center border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <HelpCircle size={18} className="text-blue-400" />
            <h3 className="font-semibold text-white text-base">Spacer Capture Protocol</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-full text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex flex-col gap-3.5 text-xs leading-relaxed">
          <div className="flex items-start gap-2.5">
            <span className="w-5 h-5 rounded-full bg-blue-600/30 text-blue-400 flex items-center justify-center font-bold text-[11px] shrink-0">
              1
            </span>
            <div>
              <strong className="text-white block font-medium">Sanitize Contact Rim</strong>
              Wipe the spacer's lower skin-contact flange with a 70% isopropyl alcohol wipe and let dry for 15 seconds.
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <span className="w-5 h-5 rounded-full bg-blue-600/30 text-blue-400 flex items-center justify-center font-bold text-[11px] shrink-0">
              2
            </span>
            <div>
              <strong className="text-white block font-medium">Place Flush Against Skin</strong>
              Rest the 3D spacer flat against the skin surface. Center the lesion inside the circular reticle without tilting.
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <span className="w-5 h-5 rounded-full bg-blue-600/30 text-blue-400 flex items-center justify-center font-bold text-[11px] shrink-0">
              3
            </span>
            <div>
              <strong className="text-white block font-medium">Uniform Lighting</strong>
              Ensure ambient room light enters through the side diffuser windows. Keep the smartphone's flash turned off.
            </div>
          </div>

          <div className="flex items-start gap-2.5">
            <span className="w-5 h-5 rounded-full bg-blue-600/30 text-blue-400 flex items-center justify-center font-bold text-[11px] shrink-0">
              4
            </span>
            <div>
              <strong className="text-white block font-medium">Tap Capture & Analyze</strong>
              Hold steady for one second while tapping Capture. The backend will normalize colour and evaluate the lesion.
            </div>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-blue-950/40 border border-blue-900/50 text-[11px] text-blue-200/90 leading-normal flex items-start gap-2">
          <Shield size={16} className="text-blue-400 shrink-0 mt-0.5" />
          <span>
            <strong>Safety Note:</strong> Suspicious lesions with irregular borders, deep pigmentation, or bleeding must receive priority specialist referral.
          </span>
        </div>

        <button
          onClick={onClose}
          className="btn-primary w-full py-2.5 text-xs font-semibold"
        >
          Got It, Continue
        </button>
      </div>
    </div>
  );
};
