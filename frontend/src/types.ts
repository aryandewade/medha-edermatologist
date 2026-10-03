export interface PredictionItem {
  class_id: string;
  label: string;
  confidence: number;
  severity: 'normal' | 'common' | 'urgent' | 'SELF_CARE' | 'ROUTINE' | 'URGENT' | string;
}

export interface Top3Item {
  class_id: string;
  label: string;
  probability: number;
  confidence_percent?: number;
}

export interface AdviceItem {
  level: 'ok' | 'uncertain' | 'urgent' | 'out_of_scope';
  title: string;
  message: string;
}

export interface QualityItem {
  blur_score: number;
  brightness: number;
  glare_ratio: number;
  reference_patch_found: boolean;
  colour_cast: string;
  status: 'good' | 'warning' | 'rejected';
  issues: string[];
}

export interface CareProtocol {
  external: string[];
  internal: string[];
  care: string[];
  urgent: string;
}

export interface PredictResponse {
  request_id: string;
  modality: string;
  engine?: 'medha_v2' | 'skinnet_ensemble' | string;
  prediction: PredictionItem;
  top3: Top3Item[];
  probabilities: Record<string, number>;
  advice: AdviceItem;
  quality: QualityItem;
  heatmap_png_base64?: string | null;
  roi_preview_png_base64?: string | null;
  model_version: string;
  latency_ms: number;
  symptom_questions?: Record<string, string>;
  candidate_diseases?: string[];
  care_protocol?: CareProtocol;
}

export interface HealthResponse {
  status: string;
  model_version: string;
  active_modality: string;
  num_classes: number;
  warmed_up: boolean;
  available_engines?: string[];
}

export interface SymptomConfirmResponse {
  confirmed_disease: string;
  severity: 'Mild' | 'Moderate' | 'Severe' | 'Out of Class' | string;
  severity_percentage: number;
  symptoms_matched: string;
  photo_confidence: number;
  confirmed_confidence: number;
  posterior_probabilities: Record<string, number>;
  is_reordered: boolean;
  original_top1: string;
  care_protocol: CareProtocol;
}

export interface HospitalItem {
  name: string;
  distance_km: number;
  latitude: number;
  longitude: number;
  location_label: string;
  maps_url: string;
}

export interface NearbyHospitalsResponse {
  location_query?: string;
  latitude?: number;
  longitude?: number;
  facilities: HospitalItem[];
}

export type Language = 'en' | 'hi' | 'mr';
