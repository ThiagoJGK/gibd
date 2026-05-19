
import { Article, UploadedImage, ImageModelId } from "./types";

export const API_BASE_URL = 'http://localhost:5001'; // Your Flask backend URL

// --- File Processing Utilities ---

/**
 * Reads a single text file and splits its content into articles.
 * Articles are assumed to be separated by one or more blank lines (two or more newlines).
 * Extracts a title from the first few words of the article.
 * @param file The .txt file to process.
 * @returns A promise that resolves to an array of Article objects.
 */
export const processTextFileToArticles = async (file: File): Promise<Article[]> => {
  return new Promise((resolve, reject) => {
    if (!file.name.endsWith('.txt') || file.type !== 'text/plain') {
      reject(new Error("Por favor, selecciona un archivo .txt válido."));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const content = reader.result as string;
      const rawArticles = content.split(/\n\s*\n/).filter(text => text.trim() !== "");
      const articles: Article[] = rawArticles.map((text, index) => {
        const trimmedContent = text.trim();
        const words = trimmedContent.split(/\s+/);
        const title = words.slice(0, 8).join(" ") + (words.length > 8 ? "..." : ""); // First 8 words as title
        return {
          id: `article-${Date.now()}-${index}-${Math.random().toString(36).substring(2, 9)}`,
          title: title || "Artículo sin título",
          content: trimmedContent,
        };
      });
      resolve(articles);
    };
    reader.onerror = (error) => {
      console.error("FileReader error:", error);
      reject(new Error(`Error al leer el archivo ${file.name}.`));
    };
    reader.readAsText(file);
  });
};


export const imageFileToUploadedImage = async (file: File): Promise<UploadedImage> => {
  return new Promise((resolve, reject) => {
    if (!file.type.startsWith('image/')) {
      reject(new Error(`El archivo ${file.name} no es una imagen válida.`));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const base64String = (reader.result as string).split(',')[1];
      resolve({ 
        id: `local-${file.name}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`, // Generate unique ID
        base64: base64String, 
        mimeType: file.type, 
        name: file.name,
        // No local embedding generation here anymore
      });
    };
    reader.onerror = (error) => {
      console.error("FileReader error:", error);
      reject(new Error(`Error al leer el archivo ${file.name}.`));
    };
    reader.readAsDataURL(file);
  });
};

// --- Simulated Embedding and Similarity (kept for potential other local uses, but not for API search) ---
const EMBEDDING_DIMENSION = 64; 

export const generateSimulatedEmbedding = (): number[] => {
  return Array.from({ length: EMBEDDING_DIMENSION }, () => Math.random() * 2 - 1);
};

export const cosineSimilarity = (vecA: number[], vecB: number[]): number => {
  if (vecA.length !== vecB.length || vecA.length === 0) {
    return 0; 
  }

  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    magnitudeA += vecA[i] * vecA[i];
    magnitudeB += vecB[i] * vecB[i];
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  if (magnitudeA === 0 || magnitudeB === 0) {
    return 0; 
  }

  return dotProduct / (magnitudeA * magnitudeB);
};

// --- API call for text search ---
export const queryTextViaAPI = async (queryText: string): Promise<(Article & { similarity: number })[]> => {
  const endpoint = `${API_BASE_URL}/api/search_text`;
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: queryText }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ message: `Error ${response.status}: ${response.statusText}` }));
      throw new Error(errorData.message || `API Error: ${response.status}`);
    }

    const data = await response.json();
    return data.results || []; 
  } catch (error) {
    console.error('API call to search_text failed:', error);
    throw error;
  }
};

// --- API call for image search ---
export const queryImageViaAPI = async (imageFile: File, modelId: ImageModelId): Promise<UploadedImage[]> => {
  const endpoint = `${API_BASE_URL}/api/search_image`;
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('model_id', modelId.toUpperCase()); // Backend expects uppercase

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData, // FormData sets Content-Type automatically for multipart/form-data
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ message: `Error ${response.status}: ${response.statusText}` }));
      throw new Error(errorData.message || `API Error: ${response.status}`);
    }
    const data = await response.json();
    // Assuming backend returns { results: [{id, name, base64, mimeType, similarity}, ...] }
    // Map to UploadedImage type
    return (data.results || []).map((img: any) => ({
        id: img.id, // API provides this ID
        name: img.name,
        base64: img.base64,
        mimeType: img.mimeType,
        similarity: img.similarity,
        modelId: modelId, // Use the modelId parameter
        // Embedding is not typically sent back for search results unless needed for further frontend processing
    }));
  } catch (error) {
    console.error('API call to search_image failed:', error);
    throw error;
  }
};