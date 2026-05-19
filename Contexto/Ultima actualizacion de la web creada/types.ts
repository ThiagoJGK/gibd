export enum Section {
  Text = 'TEXTO',
  Image = 'IMAGEN',
}

export enum AppMode {
  Consulta = 'CONSULTA',
  Mantenimiento = 'MANTENIMIENTO',
}

export enum ImageModelId {
  Tatuajes = 'TATUAJES',
  MarcasDeGanado = 'MARCAS_DE_GANADO',
  Pintura = 'PINTURA',
  CaracteresChinos = 'CARACTERES_CHINOS',
  EscudosDeFutbol = 'ESCUDOS_DE_FUTBOL',
}

export interface TextFile {
  name: string;
  content: string;
}

export interface UploadedImage {
  id: string; // Added ID property
  base64: string;
  mimeType: string;
  name: string;
  modelId?: ImageModelId; // Optional: can be useful for tracking source
  embedding?: number[]; // For simulated image embedding
  similarity?: number; // For storing similarity score during search
}

export interface Article {
  id: string;
  title: string; // For display in lists/sidebar
  content: string;
}

export interface ArticleWithEmbedding extends Article {
  embedding: number[];
  similarity?: number;
}