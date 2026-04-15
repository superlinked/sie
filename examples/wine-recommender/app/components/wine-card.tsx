import { type Wine } from "@/lib/wine-data"
import { MapPin, Calendar } from "lucide-react"
import { cn } from "@/lib/utils"

interface WineCardProps {
  wine: Wine
  compact?: boolean
}

export function WineCard({ wine, compact = false }: WineCardProps) {
  if (compact) {
    return (
      <div className="flex gap-4 rounded-xl border border-border bg-card p-4 transition-all hover:shadow-md">
        {/* Bottle Image Placeholder */}
        <div className="h-20 w-14 rounded-lg bg-gradient-to-b from-primary/20 to-primary/5 flex items-center justify-center shrink-0">
          <div className="h-14 w-4 rounded-full bg-primary/30" />
        </div>
        
        <div className="flex-1 min-w-0">
          <h4 className="font-serif text-sm font-medium text-foreground truncate">
            {wine.name}
          </h4>
          <p className="text-xs text-muted-foreground">{wine.winery}</p>
          <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {wine.vintage}
            </span>
            <span className="flex items-center gap-1">
              <MapPin className="h-3 w-3" />
              {wine.country}
            </span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={cn(
      "rounded-2xl border border-border bg-card p-6 transition-all hover:shadow-lg hover:-translate-y-0.5",
      "shadow-[0_4px_20px_-2px_rgba(114,47,55,0.08)]"
    )}>
      <div className="flex gap-5">
        {/* Bottle Image Placeholder */}
        <div className="h-32 w-20 rounded-xl bg-gradient-to-b from-primary/20 via-primary/10 to-primary/5 flex items-center justify-center shrink-0">
          <div className="h-24 w-6 rounded-full bg-gradient-to-b from-primary/40 to-primary/20" />
        </div>
        
        <div className="flex-1 space-y-3">
          <div>
            <h3 className="font-serif text-lg font-semibold text-foreground">
              {wine.name}
            </h3>
            <p className="text-sm text-muted-foreground">{wine.winery}</p>
          </div>
          
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <Calendar className="h-4 w-4" />
              {wine.vintage}
            </span>
            <span className="flex items-center gap-1.5">
              <MapPin className="h-4 w-4" />
              {wine.region}
            </span>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-border">
            <span className="text-xs px-2.5 py-1 rounded-full bg-accent/10 text-accent font-medium">
              {wine.style}
            </span>
            <span className="font-serif text-lg font-semibold text-foreground">
              ${wine.price}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
