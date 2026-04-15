import type { RecommendedWine, WineStructure } from "@/lib/wine-data"
import { WineGlassVisual } from "./wine-glass-visual"
import { MapPin, Check, Minus } from "lucide-react"
import { cn } from "@/lib/utils"

interface WineResultCardProps {
  wine: RecommendedWine
  rank: number
  userStructure: WineStructure
  userFlavors: string[]
}

const structureLabels: Record<keyof WineStructure, { label: string; icon: string }> = {
  acidity: { label: "Acidity", icon: "◇" },
  fizziness: { label: "Fizz", icon: "○" },
  intensity: { label: "Body", icon: "●" },
  sweetness: { label: "Sweet", icon: "◆" },
  tannin: { label: "Tannin", icon: "▲" },
}

export function WineResultCard({ wine, rank, userStructure, userFlavors }: WineResultCardProps) {
  // Calculate how close each attribute is to user's preference
  const getAttributeMatch = (key: keyof WineStructure): { diff: number; quality: "exact" | "close" | "far" } => {
    const diff = Math.abs(wine.structure[key] - userStructure[key])
    if (diff <= 10) return { diff, quality: "exact" }
    if (diff <= 25) return { diff, quality: "close" }
    return { diff, quality: "far" }
  }

  // Get matching flavors between wine and user selection
  const matchingFlavors = userFlavors.filter(flavor => 
    wine.flavors.some(f => f.toLowerCase() === flavor.toLowerCase())
  )

  const profileTags = [
    wine.structure.tannin >= 60 ? "Tannic" : wine.structure.tannin <= 30 ? "Smooth" : null,
    wine.structure.sweetness >= 35 ? "Sweet" : wine.structure.sweetness <= 15 ? "Dry" : null,
    wine.style || null,
  ].filter(Boolean) as string[]

  return (
    <div className={cn(
      "group relative rounded-2xl border border-border bg-card overflow-hidden transition-all duration-300",
      "hover:shadow-[0_10px_40px_-10px_rgba(114,47,55,0.15)] hover:-translate-y-1",
      rank === 1 && "ring-2 ring-secondary/50"
    )}>
      {/* Top Section with Match */}
      <div className="relative p-6 pb-4">
        {/* Rank Badge */}
        <div className={cn(
          "absolute -top-0 -left-0 h-8 w-8 rounded-br-xl flex items-center justify-center text-sm font-bold",
          rank === 1 ? "bg-secondary text-secondary-foreground" : "bg-muted text-muted-foreground"
        )}>
          {rank}
        </div>

        {/* Match Percentage Circle */}
        <div className="absolute right-4 top-4">
          <div className="relative h-12 w-12">
            <svg className="h-12 w-12 -rotate-90">
              <circle
                cx="24"
                cy="24"
                r="20"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                className="text-muted"
              />
              <circle
                cx="24"
                cy="24"
                r="20"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeDasharray={`${wine.matchPercentage * 1.26} 126`}
                className={cn(
                  wine.matchPercentage >= 90 ? "text-accent" : "text-primary"
                )}
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-xs font-bold">
              {wine.matchPercentage}%
            </span>
          </div>
        </div>

        {/* Wine Visual */}
        <div className="mx-auto mb-5 mt-2 flex h-32 items-center justify-center">
          <WineGlassVisual structure={wine.structure} className="h-28 w-24" />
        </div>

        {/* Wine Info */}
        <div className="grid gap-3 pr-14 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start sm:pr-0">
          <div className="min-w-0 space-y-1.5 text-left">
            <h4 className="font-serif text-base font-semibold leading-tight text-foreground transition-colors group-hover:text-primary">
              {wine.name}
            </h4>
            <p className="text-xs text-muted-foreground">
              {wine.vintage} · {wine.winery}
            </p>

            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <MapPin className="h-3 w-3 shrink-0" />
              <span className="truncate">{wine.region}, {wine.country}</span>
            </div>
          </div>

          {profileTags.length > 0 && (
            <div className="flex max-w-[12rem] flex-wrap gap-1.5 sm:justify-end">
              {profileTags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full bg-muted px-2.5 py-1 text-[10px] font-medium text-muted-foreground"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Structure Match Preview - Always Visible */}
      <div className="px-5 pb-3">
        <div className="flex items-center justify-center gap-1.5">
          {(Object.keys(structureLabels) as (keyof WineStructure)[]).map((key) => {
            const match = getAttributeMatch(key)
            if (key === "fizziness" && userStructure.fizziness < 10 && wine.structure.fizziness < 10) return null
            return (
              <div
                key={key}
                className={cn(
                  "h-2 w-2 rounded-full transition-colors",
                  match.quality === "exact" && "bg-accent",
                  match.quality === "close" && "bg-secondary",
                  match.quality === "far" && "bg-muted-foreground/30"
                )}
                title={`${structureLabels[key].label}: ${match.quality} match`}
              />
            )
          })}
        </div>
      </div>

      {/* Price */}
      <div className="px-5 py-3 border-t border-border">
        <span className="font-serif text-lg font-bold text-foreground">${wine.price}</span>
      </div>

      {/* Match Details - Always Visible */}
      <div className="px-5 pb-5 space-y-4 border-t border-border pt-4 bg-muted/30">
          {/* Structure Comparison */}
          <div>
            <p className="text-xs font-medium text-foreground mb-3">Structure Match</p>
            <div className="space-y-2">
              {(Object.keys(structureLabels) as (keyof WineStructure)[]).map((key) => {
                const match = getAttributeMatch(key)
                const userVal = userStructure[key]
                const wineVal = wine.structure[key]
                
                // Skip fizziness if both are essentially still
                if (key === "fizziness" && userVal < 10 && wineVal < 10) return null
                
                return (
                  <div key={key} className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground w-14">{structureLabels[key].label}</span>
                    <div className="flex-1 h-1.5 bg-muted rounded-full relative overflow-hidden">
                      {/* User preference indicator */}
                      <div 
                        className="absolute top-0 h-full w-1 bg-foreground/30 rounded-full"
                        style={{ left: `${userVal}%`, transform: "translateX(-50%)" }}
                      />
                      {/* Wine value bar */}
                      <div 
                        className={cn(
                          "absolute top-0 left-0 h-full rounded-full transition-all",
                          match.quality === "exact" && "bg-accent",
                          match.quality === "close" && "bg-secondary",
                          match.quality === "far" && "bg-muted-foreground"
                        )}
                        style={{ width: `${wineVal}%` }}
                      />
                    </div>
                    <span className="text-[10px] tabular-nums text-muted-foreground w-8">{wineVal}</span>
                    <div className={cn(
                      "h-4 w-4 rounded-full flex items-center justify-center",
                      match.quality === "exact" && "bg-accent/20 text-accent",
                      match.quality === "close" && "bg-secondary/20 text-secondary-foreground",
                      match.quality === "far" && "bg-muted text-muted-foreground"
                    )}>
                      {match.quality === "exact" ? (
                        <Check className="h-2.5 w-2.5" />
                      ) : match.quality === "close" ? (
                        <Minus className="h-2.5 w-2.5" />
                      ) : (
                        <span className="text-[8px]">~</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Flavor Match */}
          {matchingFlavors.length > 0 && (
            <div>
              <p className="text-xs font-medium text-foreground mb-2">Matching Flavors</p>
              <div className="flex flex-wrap gap-1.5">
                {matchingFlavors.map((flavor) => (
                  <span
                    key={flavor}
                    className="inline-flex items-center gap-1 rounded-full bg-accent/10 px-2 py-0.5 text-[10px] text-accent"
                  >
                    <Check className="h-2.5 w-2.5" />
                    {flavor}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Flavor Summary */}
          <div>
            <p className="text-xs font-medium text-foreground mb-1">Tasting Notes</p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {wine.flavorSummary}
            </p>
          </div>
        </div>
    </div>
  )
}
