export interface BrandInferenceResponse {
  status: string;
  query_info: {
    filename: string;
    top_k: number;
  };
  results: InferenceResult[];
}

export interface InferenceResult {
  id: string;
  name: string;
  similarity_score: number;
  distance: number;
  image_url: string;
  logo_url?: string;
  country?: string;
  league?: string;
}

export const fetchBrandInference = async (
  imageFile: File,
  topK: number = 10
): Promise<BrandInferenceResponse> => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('top_k', String(topK));

  const apiUrl =
    import.meta.env.VITE_API_URL || import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

  const response = await fetch(`${apiUrl}/api/v1/inference/siamese-brands`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Error en la inferencia: ${response.status} ${text}`);
  }

  return response.json();
};

export interface CrestInferenceResponse {
  status: string;
  query_info: {
    filename: string;
    top_k: number;
  };
  results: InferenceResult[];
}

export const fetchCrestInference = async (
  imageFile: File,
  topK: number = 10
): Promise<CrestInferenceResponse> => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('top_k', String(topK));

  const apiUrl =
    import.meta.env.VITE_API_URL || import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

  const response = await fetch(`${apiUrl}/api/v1/inference/soccer-crests`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Error en la inferencia de escudos: ${response.status} ${text}`);
  }

  return response.json();
};
