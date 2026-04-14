export type WineStructure = {
  acidity: number
  fizziness: number
  intensity: number
  sweetness: number
  tannin: number
}

export type Wine = {
  id: string
  row_index: number
  name: string
  winery: string
  vintage: string | null
  country: string | null
  region: string | null
  style: string
  price: number | null
  structure: WineStructure
  flavors: string[]
}

export type FlavorCategory = {
  category: string
  flavors: string[]
}

export type RecommendedWine = Wine & {
  matchPercentage: number
  flavorSummary: string
  reviewCount: number | null
  rerankScore: number
}

export type CatalogResponse = {
  wines: Wine[]
  flavorTags: FlavorCategory[]
}

export type RecommendationRequest = {
  structure: WineStructure
  flavors: Record<string, number>
  reference_row_indices: number[]
  top_k: number
}

export type RecommendationResponse = {
  rerank_method: string
  candidate_count: number
  results: RecommendedWine[]
}

export type DetectedWineResponse = {
  detected_wine: Wine | null
  match_score: number
  ocr_text?: string
}
