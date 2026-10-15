/**
 * GIBD 2026 — Unified Multi-Model Vision Type Contracts
 * Location: frontend/src/types/vision.ts
 *
 * Strict TypeScript typing complying with:
 * - Flat Design System OLED
 * - Zero-Cost CPU Architecture
 * - 100% Backward Compatibility with Cattle Brands Catalog (80 items)
 * - tsconfig: erasableSyntaxOnly: true & verbatimModuleSyntax: true (NO enums)
 */

import type { BrandItem } from '../data/brandsCatalog';

/**
 * Identifier for the vision sub-model / catalog.
 * String union complies with tsconfig "erasableSyntaxOnly: true" (no enums).
 */
export type VisionModelType = 'brands' | 'soccer_crests';

/**
 * Domain category for the vision item.
 */
export type VisionCategory = 'Ganadería' | 'Fútbol';

/**
 * Unified metadata contract for vision items.
 * Accommodates cattle brands (jurisdiction, description)
 * and soccer crests (league, country, foundation year).
 */
export interface VisionMetadata {
  category: VisionCategory;
  jurisdiction_or_league: string; // e.g. "Entre Ríos - Uruguay" or "Liga Profesional Argentina"
  origin_country?: string;        // e.g. "Argentina", "España", "Brasil"
  description: string;
  [key: string]: unknown;
}

/**
 * Unified result contract returned by vision inference engines
 * (both Siamese ViT-Engine and Crests ResNet/DenseNet Engine).
 */
export interface VisionMatchResult {
  id: string;
  item_code: string;
  name: string;
  filename: string;
  image_url: string;
  similarity_score: number; // Porcentaje calibrado [0.0, 100.0]
  distance: number;         // Distancia métrica euclidiana normalizada [0.0, 1.4142]
  metadata: VisionMetadata;

  // --- Campos de compatibilidad retroactiva directa con BrandItem ---
  brand_code?: string;
  jurisdiction?: string;
  description?: string;
}

/**
 * Configuration option for the vision model selector pills in Laboratorio.tsx.
 */
export interface ModelSelectorOption {
  id: VisionModelType;
  label: string;
  badge: string;
  endpoint: string;
  description: string;
  category: VisionCategory;
  latentSpace: string;
  pipelineDescription: readonly string[];
}

/**
 * Standard pre-configured options for Model Selector in Laboratorio.tsx.
 */
export const VISION_MODEL_OPTIONS: readonly ModelSelectorOption[] = [
  {
    id: 'brands',
    label: 'Marcas de Ganado',
    badge: 'CACIC 2023 / ViT',
    endpoint: '/api/v1/inference/siamese-brands',
    description: 'Cotejo biométrico y extracción de curvas sobre 80 marcas registradas en el archivo oficial GIBD.',
    category: 'Ganadería',
    latentSpace: '256-D (L2 Norm)',
    pipelineDescription: [
      'Recorte MBR + Escala Gris 224x224',
      'Vector Estructural 256-D (L2 Norm)',
      'Similitud de Coseno en CPU (<50ms)',
    ],
  },
  {
    id: 'soccer_crests',
    label: 'Escudos de Fútbol',
    badge: 'CONAISII 2023 / ResNet',
    endpoint: '/api/v1/inference/soccer-crests',
    description: 'Indexación métrica convolucional calibrada sobre 150 clubes del fútbol nacional e internacional.',
    category: 'Fútbol',
    latentSpace: '2048-D (ResNet50)',
    pipelineDescription: [
      'Entrada Cuadrada 224x224 + Pad Blanco',
      'Vector Convolucional 2048-D (L2 Norm)',
      'Calibración Sigmoidal (τ=0.82, T=0.04)',
    ],
  },
] as const;

/**
 * Reference sample item for gallery quick-testing.
 */
export interface VisionSampleItem {
  id: string;
  code: string;
  name: string;
  filename: string;
  image_url: string;
  subtitle: string;
  description: string;
  modelType: VisionModelType;
}

/**
 * State contract for the multi-model vision laboratory component.
 */
export interface VisionInferenceState {
  activeModel: VisionModelType;
  isAnalyzing: boolean;
  uploadedFile: File | null;
  previewUrl: string | null;
  cloudinaryUrl?: string | null;
  results: VisionMatchResult[];
  selectedItem: VisionMatchResult | null;
  engineSource: string | null;
  queryLatency: number | null;
  topK: number;
}

/**
 * Cloudinary API response contract for unsigned uploads.
 */
export interface CloudinaryUploadResponse {
  asset_id: string;
  public_id: string;
  version: number;
  width: number;
  height: number;
  format: string;
  resource_type: string;
  created_at: string;
  bytes: number;
  url: string;
  secure_url: string;
}

// ============================================================================
// ADAPTERS & BACKWARD COMPATIBILITY HELPERS
// ============================================================================

/**
 * Adapts an existing BrandItem into the unified VisionMatchResult contract.
 */
export function brandItemToVisionMatch(
  brand: BrandItem,
  similarityScore: number = 0,
  distance: number = 1.0
): VisionMatchResult {
  return {
    id: brand.id,
    item_code: brand.brand_code,
    name: brand.name,
    filename: brand.filename,
    image_url: brand.image_url,
    similarity_score: brand.similarity_score ?? similarityScore,
    distance: brand.distance ?? distance,
    metadata: {
      category: 'Ganadería',
      jurisdiction_or_league: brand.jurisdiction,
      origin_country: 'Argentina',
      description: brand.description,
    },
    // Direct backward compatibility:
    brand_code: brand.brand_code,
    jurisdiction: brand.jurisdiction,
    description: brand.description,
  };
}

/**
 * Adapts a VisionMatchResult back into a BrandItem (if needed by legacy functions).
 */
export function visionMatchToBrandItem(match: VisionMatchResult): BrandItem {
  return {
    id: match.id,
    brand_code: match.brand_code || match.item_code,
    name: match.name,
    filename: match.filename,
    image_url: match.image_url,
    jurisdiction: match.jurisdiction || match.metadata.jurisdiction_or_league,
    description: match.description || match.metadata.description,
    similarity_score: match.similarity_score,
    distance: match.distance,
  };
}
