export interface PredictionItem {
  class_id: string;
  label: string;
  confidence: number;
  severity: 'normal' | 'common' | 'urgent';
}

export interface Top3Item {
  class_id: string;
  label: string;
  probability: number;
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

export interface PredictResponse {
  request_id: string;
  modality: string;
  prediction: PredictionItem;
  top3: Top3Item[];
  probabilities: Record<string, number>;
  advice: AdviceItem;
  quality: QualityItem;
  heatmap_png_base64?: string | null;
  model_version: string;
  latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model_version: string;
  active_modality: string;
  num_classes: number;
  warmed_up: boolean;
}

export type Language = 'en' | 'hi' | 'mr';
