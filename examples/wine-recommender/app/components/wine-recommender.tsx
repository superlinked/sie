"use client"

import { useEffect, useRef, useState } from "react"
import { detectWineFromImage, fetchCatalog, fetchRecommendations } from "@/lib/api"
import { type FlavorCategory, type RecommendedWine, type Wine, type WineStructure } from "@/lib/wine-data"
import { WineResultCard } from "./wine-result-card"
import { WineGlassVisual } from "./wine-glass-visual"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Wine as WineIcon,
  Search,
  ArrowLeft,
  SlidersHorizontal,
  X,
  RotateCcw,
  Camera,
  Upload,
} from "lucide-react"
import { cn } from "@/lib/utils"

type View = "profile" | "results"

const defaultStructure: WineStructure = {
  acidity: 50,
  fizziness: 0,
  intensity: 50,
  sweetness: 10,
  tannin: 50,
}

const structureLabels = {
  acidity: { label: "Acidity", low: "Soft", high: "Crisp" },
  fizziness: { label: "Fizziness", low: "Still", high: "Sparkling" },
  intensity: { label: "Intensity", low: "Light", high: "Bold" },
  sweetness: { label: "Sweetness", low: "Dry", high: "Sweet" },
  tannin: { label: "Tannin", low: "Smooth", high: "Grippy" },
}

export function WineRecommender() {
  const [view, setView] = useState<View>("profile")
  const [structure, setStructure] = useState<WineStructure>(defaultStructure)
  const [selectedFlavors, setSelectedFlavors] = useState<string[]>([])
  const [selectedReferenceWine, setSelectedReferenceWine] = useState<string>("")
  const [wineSearchQuery, setWineSearchQuery] = useState("")
  const [wineSearchFocused, setWineSearchFocused] = useState(false)
  const [priceRange, setPriceRange] = useState<[number, number]>([0, 1000])
  const [sortBy, setSortBy] = useState<string>("match")
  const [selectedCountry, setSelectedCountry] = useState<string>("all")
  const [selectedStyle, setSelectedStyle] = useState<string>("all")
  const [catalogWines, setCatalogWines] = useState<Wine[]>([])
  const [flavorTags, setFlavorTags] = useState<FlavorCategory[]>([])
  const [recommendedWines, setRecommendedWines] = useState<RecommendedWine[]>([])
  const [isCatalogLoading, setIsCatalogLoading] = useState(true)
  const [isRecommendationsLoading, setIsRecommendationsLoading] = useState(false)
  const [isDetectingWine, setIsDetectingWine] = useState(false)
  const [detectedWine, setDetectedWine] = useState<Wine | null>(null)
  const [detectionMatchScore, setDetectionMatchScore] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string>("")
  const cameraInputRef = useRef<HTMLInputElement | null>(null)
  const photoLibraryInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    let mounted = true

    async function loadCatalog() {
      try {
        setIsCatalogLoading(true)
        setErrorMessage("")
        const catalog = await fetchCatalog()
        if (!mounted) return
        setCatalogWines(catalog.wines)
        setFlavorTags(catalog.flavorTags)
      } catch (error) {
        if (!mounted) return
        setErrorMessage(error instanceof Error ? error.message : "Failed to load wines from the API.")
      } finally {
        if (mounted) {
          setIsCatalogLoading(false)
        }
      }
    }

    loadCatalog()
    return () => {
      mounted = false
    }
  }, [])

  const catalogReferenceWine = catalogWines.find((wine) => wine.id === selectedReferenceWine)
  const referenceWine = catalogReferenceWine ?? (detectedWine?.id === selectedReferenceWine ? detectedWine : undefined)

  const filteredWineResults = wineSearchQuery.trim()
    ? catalogWines
        .filter((wine) =>
          `${wine.name} ${wine.winery} ${wine.region ?? ""}`
            .toLowerCase()
            .includes(wineSearchQuery.toLowerCase())
        )
        .slice(0, 8)
    : []

  const handleSelectWineFromSearch = (wineId: string) => {
    setSelectedReferenceWine(wineId)
    setDetectedWine(catalogWines.find((wine) => wine.id === wineId) ?? null)
    setDetectionMatchScore(null)
    setWineSearchQuery("")
    setWineSearchFocused(false)
  }

  useEffect(() => {
    if (!referenceWine) return

    setStructure(referenceWine.structure)
    setSelectedFlavors((prev) => [...new Set([...prev, ...referenceWine.flavors.slice(0, 4)])])
  }, [referenceWine])

  const getAllFlavors = () => {
    return flavorTags.flatMap((category) => category.flavors)
  }

  const handleStructureChange = (key: keyof WineStructure, value: number) => {
    setStructure((prev) => ({ ...prev, [key]: value }))
  }

  const handleResetStructure = () => {
    setStructure(defaultStructure)
    setSelectedReferenceWine("")
    setDetectedWine(null)
    setDetectionMatchScore(null)
  }

  const handleFlavorToggle = (flavor: string) => {
    setSelectedFlavors((prev) =>
      prev.includes(flavor)
        ? prev.filter((item) => item !== flavor)
        : [...prev, flavor]
    )
  }

  const handleFindWines = async () => {
    try {
      setIsRecommendationsLoading(true)
      setErrorMessage("")

      const flavorWeights = Object.fromEntries(selectedFlavors.map((flavor) => [flavor, 1.0]))
      const referenceRowIndices = referenceWine ? [referenceWine.row_index] : []
      const normalizedStructure = {
        acidity: structure.acidity / 100,
        fizziness: structure.fizziness / 100,
        intensity: structure.intensity / 100,
        sweetness: structure.sweetness / 100,
        tannin: structure.tannin / 100,
      }
      const response = await fetchRecommendations({
        structure: normalizedStructure,
        flavors: flavorWeights,
        reference_row_indices: referenceRowIndices,
        top_k: 12,
      })

      setRecommendedWines(response.results)
      setView("results")
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to load recommendations.")
    } finally {
      setIsRecommendationsLoading(false)
    }
  }

  const handlePickCameraImage = () => {
    cameraInputRef.current?.click()
  }

  const handlePickPhotoLibraryImage = () => {
    photoLibraryInputRef.current?.click()
  }

  const handleImageSelection = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return

    try {
      setIsDetectingWine(true)
      setErrorMessage("")
      setDetectedWine(null)
      setSelectedReferenceWine("")
      setDetectionMatchScore(null)

      const response = await detectWineFromImage(file)
      const detected = response.detected_wine
      setDetectionMatchScore(response.match_score)

      if (!detected) {
        setErrorMessage("Could not detect wine from image. Try a clearer label photo or pick the wine manually.")
        return
      }

      setDetectedWine(detected)
      setSelectedReferenceWine(detected.id)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to detect wine from image.")
    } finally {
      setIsDetectingWine(false)
    }
  }

  const filteredResults = recommendedWines
    .filter((wine) => {
      const winePrice = wine.price ?? 0
      if (winePrice < priceRange[0] || winePrice > priceRange[1]) return false
      if (selectedCountry !== "all" && wine.country !== selectedCountry) return false
      if (selectedStyle !== "all" && wine.style !== selectedStyle) return false
      return true
    })
    .sort((a, b) => {
      if (sortBy === "match") return b.matchPercentage - a.matchPercentage
      if (sortBy === "price-low") return (a.price ?? 0) - (b.price ?? 0)
      if (sortBy === "price-high") return (b.price ?? 0) - (a.price ?? 0)
      if (sortBy === "vintage") return Number(b.vintage ?? 0) - Number(a.vintage ?? 0)
      return 0
    })

  const countries = [...new Set(recommendedWines.map((wine) => wine.country).filter(Boolean))]
  const styles = [...new Set(recommendedWines.map((wine) => wine.style).filter(Boolean))]
  const hasProfile = selectedFlavors.length > 0 || JSON.stringify(structure) !== JSON.stringify(defaultStructure)

  if (view === "results") {
    return (
      <div className="min-h-screen bg-background">
        <header className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm border-b border-border">
          <div className="max-w-6xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <button
                onClick={() => setView("profile")}
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <ArrowLeft className="h-4 w-4" />
                Edit Profile
              </button>

              <h1 className="font-serif text-xl font-semibold text-foreground">
                Your Recommendations
              </h1>

              <div className="w-24" />
            </div>
          </div>
        </header>

        <div className="border-b border-border bg-card/50">
          <div className="max-w-6xl mx-auto px-6 py-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <SlidersHorizontal className="h-4 w-4" />
                <span>Filters</span>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Price</span>
                <Select value={`${priceRange[0]}-${priceRange[1]}`} onValueChange={(value) => {
                  const [min, max] = value.split("-").map(Number)
                  setPriceRange([min, max])
                }}>
                  <SelectTrigger className="h-8 w-32 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0-1000">All Prices</SelectItem>
                    <SelectItem value="0-100">Under $100</SelectItem>
                    <SelectItem value="100-200">$100 - $200</SelectItem>
                    <SelectItem value="200-500">$200 - $500</SelectItem>
                    <SelectItem value="500-1000">$500+</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Country</span>
                <Select value={selectedCountry} onValueChange={setSelectedCountry}>
                  <SelectTrigger className="h-8 w-28 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    {countries.map((country) => (
                      <SelectItem key={country} value={country ?? ""}>{country}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Style</span>
                <Select value={selectedStyle} onValueChange={setSelectedStyle}>
                  <SelectTrigger className="h-8 w-36 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Styles</SelectItem>
                    {styles.map((style) => (
                      <SelectItem key={style} value={style}>{style}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex-1" />

              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Sort by</span>
                <Select value={sortBy} onValueChange={setSortBy}>
                  <SelectTrigger className="h-8 w-36 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="match">Best Match</SelectItem>
                    <SelectItem value="price-low">Price: Low to High</SelectItem>
                    <SelectItem value="price-high">Price: High to Low</SelectItem>
                    <SelectItem value="vintage">Vintage</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        </div>

        <div className="border-b border-border bg-muted/30">
          <div className="max-w-6xl mx-auto px-6 py-3">
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <span className="text-muted-foreground">Based on:</span>
              {referenceWine && (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-3 py-1 text-primary">
                  <WineIcon className="h-3 w-3" />
                  {referenceWine.name}
                </span>
              )}
              {selectedFlavors.slice(0, 4).map((flavor) => (
                <span
                  key={flavor}
                  className="inline-flex items-center rounded-full bg-accent/10 px-3 py-1 text-accent text-xs"
                >
                  {flavor}
                </span>
              ))}
              {selectedFlavors.length > 4 && (
                <span className="text-xs text-muted-foreground">
                  +{selectedFlavors.length - 4} more
                </span>
              )}
            </div>
          </div>
        </div>

        <main className="max-w-6xl mx-auto px-6 py-8">
          {errorMessage && (
            <div className="mb-6 rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
              {errorMessage}
            </div>
          )}

          <p className="text-sm text-muted-foreground mb-6">
            {filteredResults.length} wines found
          </p>

          {filteredResults.length > 0 ? (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {filteredResults.map((wine, index) => (
                <WineResultCard
                  key={wine.id}
                  wine={wine}
                  rank={index + 1}
                  userStructure={structure}
                  userFlavors={selectedFlavors}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <p className="text-muted-foreground">No wines match your filters.</p>
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => {
                  setPriceRange([0, 1000])
                  setSelectedCountry("all")
                  setSelectedStyle("all")
                }}
              >
                Clear Filters
              </Button>
            </div>
          )}
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="max-w-5xl mx-auto px-6 py-6 text-center">
          <h1 className="font-serif text-2xl font-semibold text-foreground">
            Build Your Wine
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Adjust the sliders to craft your ideal profile
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {errorMessage && (
          <div className="mb-6 rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {errorMessage}
          </div>
        )}

        <div className="grid lg:grid-cols-[340px,minmax(0,1fr)] gap-8 items-start">
          <div className="lg:sticky lg:top-8 lg:self-start space-y-6">
            <div className="rounded-2xl border border-border bg-card p-6">
              <h3 className="text-sm font-medium text-foreground text-center mb-2">
                Your Wine
              </h3>

              <WineGlassVisual structure={structure} className="h-[260px]" />

              <div className="mt-4 pt-4 border-t border-border space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Body</span>
                  <span className="text-foreground">
                    {structure.intensity > 70 ? "Full" : structure.intensity > 40 ? "Medium" : "Light"}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Style</span>
                  <span className="text-foreground">
                    {structure.fizziness > 50 ? "Sparkling" : structure.sweetness > 30 ? "Off-dry" : "Dry"}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Character</span>
                  <span className="text-foreground">
                    {structure.tannin > 60 ? "Structured" : structure.acidity > 60 ? "Fresh" : "Smooth"}
                  </span>
                </div>
              </div>
            </div>

            <Button
              onClick={handleFindWines}
              disabled={!hasProfile || isRecommendationsLoading || isCatalogLoading}
              className="w-full h-12"
              size="lg"
            >
              <Search className="mr-2 h-4 w-4" />
              {isRecommendationsLoading ? "Finding wines..." : "Find Wines"}
            </Button>
          </div>

          <div className="space-y-6 min-w-0">
            <div className="rounded-2xl border border-border bg-card p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-foreground">
                  Start from a wine you love
                </h3>
                {selectedReferenceWine && (
                  <button
                    onClick={() => {
                      setSelectedReferenceWine("")
                      setDetectedWine(null)
                      setDetectionMatchScore(null)
                    }}
                    className="text-xs text-muted-foreground hover:text-foreground"
                  >
                    Clear
                  </button>
                )}
              </div>

              <input
                ref={cameraInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={handleImageSelection}
              />

              <input
                ref={photoLibraryInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageSelection}
              />

              <div className="mb-4 grid gap-3 sm:grid-cols-2">
                <button
                  onClick={handlePickCameraImage}
                  disabled={isCatalogLoading || isDetectingWine}
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-primary/30 bg-primary/5 px-4 py-3 text-sm text-foreground transition-colors hover:bg-primary/10 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isDetectingWine ? (
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  ) : (
                    <Camera className="h-4 w-4 text-primary" />
                  )}
                  <span>{isDetectingWine ? "Detecting..." : "Take photo"}</span>
                </button>

                <button
                  onClick={handlePickPhotoLibraryImage}
                  disabled={isCatalogLoading || isDetectingWine}
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 py-3 text-sm text-foreground transition-colors hover:bg-muted/50 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <Upload className="h-4 w-4 text-primary" />
                  <span>Upload image</span>
                </button>
              </div>

              {(detectedWine || detectionMatchScore !== null) && (
                <div className="mb-4 rounded-xl border border-accent/20 bg-accent/5 px-4 py-3 text-sm">
                  {detectedWine ? (
                    <p className="font-medium text-foreground">
                      {detectedWine.name} {detectedWine.vintage ? `· ${detectedWine.vintage}` : ""}
                    </p>
                  ) : null}
                  {detectionMatchScore !== null && (
                    <p className={cn("text-xs text-muted-foreground", detectedWine && "mt-1")}>
                      Match score: {Math.round(detectionMatchScore * 100)}%
                    </p>
                  )}
                </div>
              )}

              {referenceWine && (
                <div className="mb-4 flex items-center gap-3 rounded-xl border border-primary/10 bg-primary/5 p-3">
                  <div className="w-10 h-14 rounded bg-gradient-to-b from-primary/20 to-primary/40 flex items-center justify-center">
                    <WineIcon className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">{referenceWine.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {referenceWine.vintage} · {referenceWine.region}
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setSelectedReferenceWine("")
                      setDetectedWine(null)
                      setDetectionMatchScore(null)
                    }}
                    className="p-1.5 rounded-full hover:bg-muted transition-colors"
                  >
                    <X className="h-4 w-4 text-muted-foreground" />
                  </button>
                </div>
              )}

              <div className="relative">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    value={wineSearchQuery}
                    onChange={(e) => setWineSearchQuery(e.target.value)}
                    onFocus={() => setWineSearchFocused(true)}
                    onBlur={() => setTimeout(() => setWineSearchFocused(false), 200)}
                    placeholder={isCatalogLoading ? "Loading wines..." : "Search wines..."}
                    className="h-11 pl-10 bg-background"
                    disabled={isCatalogLoading}
                  />
                </div>

                {wineSearchFocused && wineSearchQuery.trim() && (
                  <div className="absolute top-full left-0 right-0 mt-2 bg-card border border-border rounded-xl shadow-lg overflow-hidden z-20">
                    {filteredWineResults.length > 0 ? (
                      <div className="max-h-64 overflow-y-auto">
                        {filteredWineResults.map((wine) => (
                          <button
                            key={wine.id}
                            onClick={() => handleSelectWineFromSearch(wine.id)}
                            className="w-full flex items-center gap-3 p-3 hover:bg-muted/50 transition-colors text-left"
                          >
                            <div className="w-8 h-10 rounded bg-gradient-to-b from-primary/10 to-primary/30 flex items-center justify-center shrink-0">
                              <WineIcon className="h-3.5 w-3.5 text-primary/60" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium text-foreground truncate">{wine.name}</p>
                              <p className="text-xs text-muted-foreground truncate">
                                {wine.vintage} · {wine.region}
                              </p>
                            </div>
                            <span className="text-xs text-muted-foreground">${wine.price ?? "-"}</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="p-4 text-center text-sm text-muted-foreground">
                        No wines found
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-2xl border border-border bg-card p-5">
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-sm font-medium text-foreground">Structure</h3>
                <button
                  onClick={handleResetStructure}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  <RotateCcw className="h-3 w-3" />
                  Reset
                </button>
              </div>

              <div className="space-y-5">
                {(Object.keys(structureLabels) as (keyof WineStructure)[]).map((key) => (
                  <div key={key} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground font-medium">
                        {structureLabels[key].label}
                      </span>
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {structure[key]}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground w-16">
                        {structureLabels[key].low}
                      </span>
                      <div className="relative flex-1 h-6 flex items-center">
                        <div className="absolute inset-x-0 h-2 bg-muted rounded-full" />
                        <div
                          className="absolute left-0 h-2 bg-primary/30 rounded-full transition-all duration-200"
                          style={{ width: `${structure[key]}%` }}
                        />
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={structure[key]}
                          onChange={(e) => handleStructureChange(key, Number(e.target.value))}
                          className="slider-wine w-full"
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-16 text-right">
                        {structureLabels[key].high}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-border bg-card p-5">
              <h3 className="text-sm font-medium text-foreground mb-4">Flavors</h3>

              <div className="max-h-[540px] overflow-y-auto pr-1 space-y-4">
                {flavorTags.map((category) => (
                  <div key={category.category}>
                    <p className="text-xs text-muted-foreground mb-2">{category.category}</p>
                    <div className="flex flex-wrap gap-2">
                      {category.flavors.map((flavor) => (
                        <button
                          key={flavor}
                          onClick={() => handleFlavorToggle(flavor)}
                          className={cn(
                            "px-3 py-1.5 rounded-full text-xs transition-all",
                            selectedFlavors.includes(flavor)
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted text-muted-foreground hover:bg-muted/80"
                          )}
                        >
                          {flavor}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {selectedFlavors.length > 0 && (
                <div className="mt-5 pt-4 border-t border-border">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-xs text-muted-foreground">Selected ({selectedFlavors.length})</p>
                    <button
                      onClick={() => setSelectedFlavors([])}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      Clear all
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {selectedFlavors.map((flavor) => (
                      <span
                        key={flavor}
                        className="inline-flex items-center gap-1.5 rounded-full bg-accent text-accent-foreground px-3 py-1.5 text-xs"
                      >
                        {flavor}
                        <button
                          onClick={() => handleFlavorToggle(flavor)}
                          className="hover:opacity-70"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}
