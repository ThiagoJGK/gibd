/**
 * GIBD 2026 — Latent Space & Dimensionality Reduction Type Contracts (G5)
 * Location: frontend/src/types/projection.ts
 *
 * Strict TypeScript typing complying with:
 * - Flat Design System OLED
 * - Zero-Cost CPU Architecture
 * - tsconfig: erasableSyntaxOnly: true & verbatimModuleSyntax: true (NO enums)
 * - Research Group G5: Dimensionality Reduction & Vector Visualization
 */

export type ProjectionDataset = 'brands' | 'soccer_crests' | 'tattoos';

export type ProjectionReductionMethod = 'PCA' | 'UMAP';

export type ProjectionDimension = 2 | 3;

/**
 * Representación de un nodo proyectado en el espacio latente.
 */
export interface LatentNode {
  id: string;
  item_code: string;
  label: string;
  category: string;
  thumbnail_url: string;
  /** Coordenadas [x, y, z] normalizadas en el rango [-1.0, 1.0] */
  coordinates_3d: [number, number, number];
  /** Coordenadas [x, y] normalizadas en el rango [-1.0, 1.0] */
  coordinates_2d: [number, number];
  /** Identificador de cluster asignado por K-Means o DBSCAN */
  cluster_id: number;
  metadata?: Record<string, unknown>;
}

/**
 * Payload completo devuelto por el endpoint de proyección /api/v1/embeddings/projection
 */
export interface LatentSpacePayload {
  dataset: ProjectionDataset;
  reduction_method: ProjectionReductionMethod;
  total_nodes: number;
  /** Varianza explicada por cada componente principal en PCA */
  variance_explained_ratio?: number[];
  nodes: LatentNode[];
  /** Nodo de consulta proyectado en tiempo real (si el usuario subió una imagen) */
  query_node?: LatentNode;
}

/**
 * Solicitud para proyectar un vector de consulta al vuelo
 */
export interface ProjectPointRequest {
  dataset: ProjectionDataset;
  reduction_method?: ProjectionReductionMethod;
  vector?: number[];
  image_url?: string;
}

/**
 * Respuesta de proyección de un punto individual
 */
export interface ProjectPointResponse {
  coordinates_3d: [number, number, number];
  coordinates_2d: [number, number];
  nearest_cluster_id?: number;
  nearest_neighbors?: Array<{
    id: string;
    label: string;
    distance: number;
  }>;
}

/**
 * Opciones preconfiguradas para selector de datasets 3D
 */
export interface Dataset3DOption {
  id: ProjectionDataset;
  label: string;
  badge: string;
  description: string;
  totalItems: number;
  latentDimensions: string;
}

export const DATASET_3D_OPTIONS: readonly Dataset3DOption[] = [
  {
    id: 'brands',
    label: 'Marcas de Ganado',
    badge: 'CACIC 2023 / ViT',
    description: 'Nube de 80 marcas vacunas proyectadas desde tensores estructurales 256-D.',
    totalItems: 80,
    latentDimensions: '256-D -> 3-D',
  },
  {
    id: 'soccer_crests',
    label: 'Escudos de Fútbol',
    badge: 'CONAISII 2023 / ResNet',
    description: 'Nube de 150 clubes proyectados desde espacios latentes convolucionales 256-D.',
    totalItems: 150,
    latentDimensions: '256-D -> 3-D',
  },
  {
    id: 'tattoos',
    label: 'Detección de Tatuajes',
    badge: 'GIBD Vault / Experimental',
    description: 'Espacio latente de patrones biométricos dérmicos.',
    totalItems: 64,
    latentDimensions: '512-D -> 3-D',
  },
];
