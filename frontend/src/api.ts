import {
  HealthResponse,
  PredictResponse,
  Language,
  SymptomConfirmResponse,
  NearbyHospitalsResponse
} from './types';

function getApiBase(): string {
  const envUrl = (import.meta.env.VITE_API_URL || '').trim();
  if (!envUrl) return '/api/v1';
  // Strip trailing slashes
  const clean = envUrl.replace(/\/+$/, '');
  // Ensure it points to the /api/v1 prefix
  return clean.endsWith('/api/v1') ? clean : `${clean}/api/v1`;
}

const API_BASE = getApiBase();

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, {
    headers: { 'Accept': 'application/json' }
  });
  if (!res.ok) {
    throw new Error(`Health check failed with HTTP ${res.status}`);
  }
  return res.json();
}

export async function predictImage(
  imageBlob: Blob,
  lang: Language = 'en',
  heatmap: boolean = true,
  engine: 'medha_v2' | 'skinnet_ensemble' = 'medha_v2'
): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('image', imageBlob, 'capture.jpg');
  formData.append('modality', 'mobile_spacer');
  formData.append('lang', lang);
  formData.append('heatmap', heatmap ? 'true' : 'false');
  formData.append('capture_height_mm', '35.0');
  formData.append('engine', engine);

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    let errorDetail = `Prediction failed with status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
      else if (errJson.error?.message) errorDetail = errJson.error.message;
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return res.json();
}

export async function confirmSymptoms(
  diseases: string[],
  answers: Record<string, string | boolean>,
  probabilities?: number[]
): Promise<SymptomConfirmResponse> {
  const res = await fetch(`${API_BASE}/symptoms/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      diseases,
      answers,
      probabilities
    })
  });

  if (!res.ok) {
    let errorDetail = `Symptom confirmation failed with status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return res.json();
}

export async function fetchNearbyHospitals(
  location?: string,
  latitude?: number,
  longitude?: number
): Promise<NearbyHospitalsResponse> {
  const res = await fetch(`${API_BASE}/hospitals/nearby`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      location: location || undefined,
      latitude: latitude || undefined,
      longitude: longitude || undefined
    })
  });

  if (!res.ok) {
    let errorDetail = `Hospital search failed with status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return res.json();
}

export async function downloadReferralPdf(
  result: PredictResponse,
  patientRef: string = "Screening Case"
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result, patient_ref: patientRef })
    });

    if (!res.ok) {
      throw new Error(`PDF generation endpoint returned status ${res.status}`);
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `EDerm_Referral_${result.request_id || 'case'}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  } catch (err) {
    console.warn("Falling back to window.print for PDF:", err);
    window.print();
  }
}

/** Mock fallback for demo if backend is offline */
export function generateMockPrediction(type: 'eczema' | 'suspicious' | 'healthy'): PredictResponse {
  if (type === 'suspicious') {
    return {
      request_id: 'mock-suspicious-01',
      modality: 'mobile_spacer',
      prediction: {
        class_id: 'suspicious_lesion',
        label: 'Suspicious Lesion',
        confidence: 0.784,
        severity: 'urgent'
      },
      top3: [
        { class_id: 'suspicious_lesion', label: 'Suspicious Lesion', probability: 0.784 },
        { class_id: 'psoriasis', label: 'Psoriasis', probability: 0.121 },
        { class_id: 'eczema', label: 'Eczema', probability: 0.052 }
      ],
      probabilities: {
        suspicious_lesion: 0.784,
        psoriasis: 0.121,
        eczema: 0.052,
        tinea: 0.021,
        acne: 0.012,
        healthy: 0.010
      },
      advice: {
        level: 'urgent',
        title: 'Specialist Review Recommended',
        message: 'Atypical morphological features detected. Facilitate prompt in-person dermatologist examination.'
      },
      quality: {
        blur_score: 142.0,
        brightness: 120.0,
        glare_ratio: 0.02,
        reference_patch_found: true,
        colour_cast: 'corrected_neutral',
        status: 'good',
        issues: []
      },
      model_version: 'v2.0-mobile-effnetb0',
      latency_ms: 32.0
    };
  }

  if (type === 'healthy') {
    return {
      request_id: 'mock-healthy-01',
      modality: 'mobile_spacer',
      prediction: {
        class_id: 'healthy',
        label: 'Healthy Skin',
        confidence: 0.932,
        severity: 'normal'
      },
      top3: [
        { class_id: 'healthy', label: 'Healthy Skin', probability: 0.932 },
        { class_id: 'eczema', label: 'Eczema', probability: 0.038 },
        { class_id: 'acne', label: 'Acne', probability: 0.015 }
      ],
      probabilities: {
        healthy: 0.932,
        eczema: 0.038,
        acne: 0.015,
        psoriasis: 0.008,
        tinea: 0.005,
        suspicious_lesion: 0.002
      },
      advice: {
        level: 'ok',
        title: 'Normal Skin Appearance',
        message: 'No active inflammatory or suspicious neoplastic lesions detected on skin surface.'
      },
      quality: {
        blur_score: 165.0,
        brightness: 135.0,
        glare_ratio: 0.01,
        reference_patch_found: true,
        colour_cast: 'corrected_neutral',
        status: 'good',
        issues: []
      },
      model_version: 'v2.0-mobile-effnetb0',
      latency_ms: 28.5
    };
  }

  // Default Eczema
  return {
    request_id: 'mock-eczema-01',
    modality: 'mobile_spacer',
    prediction: {
      class_id: 'eczema',
      label: 'Eczema',
      confidence: 0.856,
      severity: 'common'
    },
    top3: [
      { class_id: 'eczema', label: 'Eczema', probability: 0.856 },
      { class_id: 'psoriasis', label: 'Psoriasis', probability: 0.089 },
      { class_id: 'tinea', label: 'Tinea (Ringworm)', probability: 0.031 }
    ],
    probabilities: {
      eczema: 0.856,
      psoriasis: 0.089,
      tinea: 0.031,
      acne: 0.012,
      healthy: 0.008,
      suspicious_lesion: 0.004
    },
    advice: {
      level: 'ok',
      title: 'Features Consistent with Eczema',
      message: 'Presentation aligns with atopic dermatitis / eczema. Standard primary care management recommended.'
    },
    quality: {
      blur_score: 138.0,
      brightness: 122.0,
      glare_ratio: 0.02,
      reference_patch_found: true,
      colour_cast: 'corrected_neutral',
      status: 'good',
      issues: []
    },
    model_version: 'v2.0-mobile-effnetb0',
    latency_ms: 34.0
  };
}
