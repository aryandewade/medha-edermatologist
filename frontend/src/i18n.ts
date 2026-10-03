import { Language } from './types';

export interface Translations {
  appTitle: string;
  appSubtitle: string;
  spacerBadge: string;
  captureBtn: string;
  analyzingBtn: string;
  uploadPhoto: string;
  instantDemos: string;
  demoHint: string;
  alignSpacer: string;
  focusOk: string;
  lightOk: string;
  newCapture: string;
  exportPdf: string;
  primaryFinding: string;
  confidence: string;
  analyzedFrame: string;
  heatmapOn: string;
  heatmapOff: string;
  differentialHeading: string;
  viewAllProbs: string;
  hideAllProbs: string;
  opticalMetrics: string;
  statusGood: string;
  howItWorksTitle: string;
  howItWorksBody: string;
  disclaimer: string;
  footerHospital: string;
  conditions: Record<string, string>;
  triageHeaders: {
    urgent: string;
    uncertain: string;
    normal: string;
    common: string;
  };
}

export const translations: Record<Language, Translations> = {
  en: {
    appTitle: "E-Dermatologist",
    appSubtitle: "Clinical Skin Screening Aid",
    spacerBadge: "3D Spacer",
    captureBtn: "CAPTURE & ANALYZE",
    analyzingBtn: "Analyzing Frame & Colour...",
    uploadPhoto: "📁 Upload photo",
    instantDemos: "Instant Demo Cases",
    demoHint: "Test without camera",
    alignSpacer: "Align 3D Spacer Flat",
    focusOk: "Focus OK",
    lightOk: "Light OK",
    newCapture: "New Capture",
    exportPdf: "Export Referral PDF",
    primaryFinding: "Primary Screening Finding",
    confidence: "Confidence",
    analyzedFrame: "Analyzed Skin Frame",
    heatmapOn: "Heatmap: ON",
    heatmapOff: "Heatmap: OFF",
    differentialHeading: "Differential Distribution (Top-3)",
    viewAllProbs: "View all 6 class probabilities",
    hideAllProbs: "Hide full probability vector",
    opticalMetrics: "Digital Optical Assessment",
    statusGood: "STATUS: GOOD",
    howItWorksTitle: "Standardized 3D Spacer Ingestion:",
    howItWorksBody: "Focal height is locked to 35 mm to prevent blur. Images undergo automated Shades-of-Gray colour constancy and central lesion localization before EfficientNet-B0 inference.",
    disclaimer: "Screening decision-support aid only. Not a medical diagnosis. All suspicious or non-responsive skin conditions must be formally evaluated by a qualified dermatologist.",
    footerHospital: "K J Somaiya College of Engineering × Kaushalya Hospital",
    conditions: {
      eczema: "Eczema (Atopic Dermatitis)",
      psoriasis: "Psoriasis",
      tinea: "Tinea (Ringworm)",
      acne: "Acne Vulgaris",
      healthy: "Healthy Skin",
      suspicious_lesion: "Suspicious Lesion"
    },
    triageHeaders: {
      urgent: "🚨 URGENT REFERRAL RECOMMENDED",
      uncertain: "⚠️ UNCERTAIN / DOCTOR REVIEW",
      normal: "✓ NORMAL SKIN PRESENTATION",
      common: "PRIMARY CARE SCREENING"
    }
  },

  hi: {
    appTitle: "ई-डर्मेटोलॉजिस्ट",
    appSubtitle: "त्वचा रोग स्क्रीनिंग सहायता",
    spacerBadge: "3D स्पेसर",
    captureBtn: "फोटो लें और विश्लेषण करें",
    analyzingBtn: "विश्लेषण जारी है...",
    uploadPhoto: "📁 गैलरी से फोटो अपलोड करें",
    instantDemos: "डेमो टेस्ट केस",
    demoHint: "कैमरा के बिना परीक्षण करें",
    alignSpacer: "स्पेसर को त्वचा पर सीधा रखें",
    focusOk: "फोकस सही",
    lightOk: "प्रकाश सही",
    newCapture: "नई जांच",
    exportPdf: "रेफरल रिपोर्ट डाउनलोड करें",
    primaryFinding: "मुख्य स्क्रीनिंग परिणाम",
    confidence: "सटीकता / विश्वास",
    analyzedFrame: "विश्लेषित त्वचा चित्र",
    heatmapOn: "हीटमैप: चालू",
    heatmapOff: "हीटमैप: बंद",
    differentialHeading: "शीर्ष-3 संभावित स्थितियां",
    viewAllProbs: "सभी 6 बीमारियों की संभावना देखें",
    hideAllProbs: "संभावना सूची छुपाएं",
    opticalMetrics: "ऑप्टिकल गुणवत्ता मूल्यांकन",
    statusGood: "स्थिति: उत्तम",
    howItWorksTitle: "3D स्पेसर तकनीक:",
    howItWorksBody: "35 मिमी की दूरी से धुंधलापन रोका जाता है। शेड्स-ऑफ-ग्रे कलर सुधार और घाव पहचान के बाद AI द्वारा जांच की जाती है।",
    disclaimer: "यह केवल प्राथमिक स्क्रीनिंग सहायता है, अंतिम चिकित्सा निदान नहीं। संदिग्ध या गंभीर मामलों में तुरंत त्वचा विशेषज्ञ से परामर्श लें।",
    footerHospital: "के जे सोमैया कॉलेज ऑफ इंजीनियरिंग × कौशल्या हॉस्पिटल",
    conditions: {
      eczema: "एक्जिमा (एटोपिक डर्मेटाइटिस)",
      psoriasis: "सोरायसिस",
      tinea: "दाद (रिंगवर्म / फंगल)",
      acne: "मुंहासे (एक्ने)",
      healthy: "स्वस्थ त्वचा",
      suspicious_lesion: "संदिग्ध त्वचा घाव (तत्काल रेफरल)"
    },
    triageHeaders: {
      urgent: "🚨 तत्काल विशेषज्ञ रेफरल अनुशंसित",
      uncertain: "⚠️ अनिश्चित / डॉक्टर से परामर्श लें",
      normal: "✓ सामान्य त्वचा की स्थिति",
      common: "प्राथमिक स्वास्थ्य केंद्र प्रबंधन"
    }
  },

  mr: {
    appTitle: "ई-डर्मेटोलॉजिस्ट",
    appSubtitle: "त्वचारोग तपासणी सहाय्यक",
    spacerBadge: "3D स्पेसर",
    captureBtn: "फोटो काढा आणि तपासा",
    analyzingBtn: "तपासणी सुरू आहे...",
    uploadPhoto: "📁 गॅलरीतून फोटो अपलोड करा",
    instantDemos: "डेमो चाचणी नमुने",
    demoHint: "कॅमेऱ्याशिवाय चाचणी करा",
    alignSpacer: "स्पेसर त्वचेवर सपाट ठेवा",
    focusOk: "फोकस योग्य",
    lightOk: "प्रकाश योग्य",
    newCapture: "नवीन तपासणी",
    exportPdf: "रेफरल रिपोर्ट डाउनलोड करा",
    primaryFinding: "प्राथमिक तपासणी निष्कर्ष",
    confidence: "विश्वासार्हता टक्केवारी",
    analyzedFrame: "तपासलेली त्वचेची प्रतिमा",
    heatmapOn: "हीटमॅप: चालू",
    heatmapOff: "हीटमॅप: बंद",
    differentialHeading: "शीर्ष-3 संभाव्य आजार",
    viewAllProbs: "सर्व 6 आजारांची शक्यता पहा",
    hideAllProbs: "शक्यता यादी लपवा",
    opticalMetrics: "ऑप्टिकल दर्जा तपासणी",
    statusGood: "दर्जा: उत्तम",
    howItWorksTitle: "3D स्पेसर तंत्रज्ञान:",
    howItWorksBody: "35 मिमी अंतरामुळे अस्पष्टता टाळली जाते. शेड्स-ऑफ-ग्रे रंग सुधारणा आणि जखम ओळखून AI द्वारे तपासणी केली जाते.",
    disclaimer: "हे केवळ प्राथमिक तपासणी साधन आहे, वैद्यकीय निदान नाही. संशयास्पद किंवा गंभीर त्वचा विकारांसाठी त्वरित तज्ज्ञ डॉक्टरांचा सल्ला घ्या.",
    footerHospital: "के जे सोमय्या कॉलेज ऑफ इंजिनीअरिंग × कौशल्या हॉस्पिटल",
    conditions: {
      eczema: "एक्झिमा (खाज व कोरडी त्वचा)",
      psoriasis: "सोरायसिस (त्वचेवरील चट्टे)",
      tinea: "दाद (बुरशीजन्य संसर्ग)",
      acne: "मुरुमे (पिंपल्स)",
      healthy: "निरोगी त्वचा",
      suspicious_lesion: "संशयास्पद जखम (त्वरीत तपासणी)"
    },
    triageHeaders: {
      urgent: "🚨 त्वरीत तज्ज्ञ तपासणीची शिफारस",
      uncertain: "⚠️ अस्पष्ट / डॉक्टरांचा सल्ला घ्या",
      normal: "✓ निरोगी त्वचा आढळली",
      common: "प्राथमिक आरोग्य केंद्र उपचार"
    }
  }
};
