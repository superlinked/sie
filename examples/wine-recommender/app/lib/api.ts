import type {
  CatalogResponse,
  DetectedWineResponse,
  RecommendationRequest,
  RecommendationResponse,
} from "@/lib/wine-data"

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000"

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || `Request failed with status ${response.status}`)
  }

  return response.json() as Promise<T>
}

export async function fetchCatalog(): Promise<CatalogResponse> {
  const response = await fetch(`${API_BASE_URL}/catalog`, {
    cache: "no-store",
  })
  return parseResponse<CatalogResponse>(response)
}

export async function fetchRecommendations(
  payload: RecommendationRequest,
): Promise<RecommendationResponse> {
  const response = await fetch(`${API_BASE_URL}/recommendations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })

  return parseResponse<RecommendationResponse>(response)
}

export async function detectWineFromImage(file: File): Promise<DetectedWineResponse> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${API_BASE_URL}/detect-wine-image`, {
    method: "POST",
    body: formData,
  })

  return parseResponse<DetectedWineResponse>(response)
}
