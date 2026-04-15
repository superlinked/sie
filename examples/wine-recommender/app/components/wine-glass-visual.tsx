"use client"

import { useMemo } from "react"
import { cn } from "@/lib/utils"
import type { WineStructure } from "@/lib/wine-data"

interface WineGlassVisualProps {
  structure: WineStructure
  className?: string
}

// Linear interpolation helper
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

export function WineGlassVisual({ structure, className }: WineGlassVisualProps) {
  const { acidity, fizziness, intensity, sweetness, tannin } = structure
  
  // Morph factor (0 = wine glass, 1 = champagne flute)
  const morphFactor = fizziness / 100

  // Generate interpolated glass path
  const glassPath = useMemo(() => {
    // Wine glass control points
    const wineGlass = {
      topLeftX: 25,
      topRightX: 95,
      bowlLeftX: 15,
      bowlRightX: 105,
      bowlBottomY: 100,
      neckLeftX: 45,
      neckRightX: 75,
      stemTopY: 105,
      stemBottomY: 140,
      baseLeftX: 35,
      baseRightX: 85,
      baseY: 150,
      bowlCurveY: 70,
    }
    
    // Champagne flute control points (taller, narrower)
    const flute = {
      topLeftX: 40,
      topRightX: 80,
      bowlLeftX: 38,
      bowlRightX: 82,
      bowlBottomY: 110,
      neckLeftX: 50,
      neckRightX: 70,
      stemTopY: 115,
      stemBottomY: 150,
      baseLeftX: 40,
      baseRightX: 80,
      baseY: 160,
      bowlCurveY: 60,
    }
    
    // Interpolate all values
    const topLeftX = lerp(wineGlass.topLeftX, flute.topLeftX, morphFactor)
    const topRightX = lerp(wineGlass.topRightX, flute.topRightX, morphFactor)
    const bowlLeftX = lerp(wineGlass.bowlLeftX, flute.bowlLeftX, morphFactor)
    const bowlRightX = lerp(wineGlass.bowlRightX, flute.bowlRightX, morphFactor)
    const bowlBottomY = lerp(wineGlass.bowlBottomY, flute.bowlBottomY, morphFactor)
    const neckLeftX = lerp(wineGlass.neckLeftX, flute.neckLeftX, morphFactor)
    const neckRightX = lerp(wineGlass.neckRightX, flute.neckRightX, morphFactor)
    const stemTopY = lerp(wineGlass.stemTopY, flute.stemTopY, morphFactor)
    const stemBottomY = lerp(wineGlass.stemBottomY, flute.stemBottomY, morphFactor)
    const baseLeftX = lerp(wineGlass.baseLeftX, flute.baseLeftX, morphFactor)
    const baseRightX = lerp(wineGlass.baseRightX, flute.baseRightX, morphFactor)
    const baseY = lerp(wineGlass.baseY, flute.baseY, morphFactor)
    const bowlCurveY = lerp(wineGlass.bowlCurveY, flute.bowlCurveY, morphFactor)
    
    return {
      path: `M${topLeftX} 20 Q${bowlLeftX} ${bowlCurveY} ${neckLeftX} ${bowlBottomY} L${neckLeftX} ${stemTopY} L${neckLeftX} ${stemBottomY} L${baseLeftX} ${baseY - 5} L${baseLeftX} ${baseY} L${baseRightX} ${baseY} L${baseRightX} ${baseY - 5} L${neckRightX} ${stemBottomY} L${neckRightX} ${stemTopY} L${neckRightX} ${bowlBottomY} Q${bowlRightX} ${bowlCurveY} ${topRightX} 20 Z`,
      clipPath: `M${topLeftX} 20 Q${bowlLeftX} ${bowlCurveY} ${neckLeftX} ${bowlBottomY} L${neckLeftX} ${stemTopY} L${neckLeftX} ${stemBottomY} L${baseLeftX} ${baseY - 5} L${baseLeftX} ${baseY} L${baseRightX} ${baseY} L${baseRightX} ${baseY - 5} L${neckRightX} ${stemBottomY} L${neckRightX} ${stemTopY} L${neckRightX} ${bowlBottomY} Q${bowlRightX} ${bowlCurveY} ${topRightX} 20 Z`,
      fillTop: bowlBottomY,
      rimWidth: (topRightX - topLeftX) / 2,
      centerX: (topLeftX + topRightX) / 2,
      baseY,
    }
  }, [morphFactor])

  // Wine color based on intensity and tannin
  const getWineColor = () => {
    const darkness = (intensity + tannin) / 2
    if (fizziness > 50) {
      // Sparkling - golden tones
      return `hsl(45, ${60 + sweetness * 0.3}%, ${85 - darkness * 0.3}%)`
    }
    if (darkness > 70) {
      return `hsl(${350 - tannin * 0.1}, ${70 + intensity * 0.2}%, ${25 + (100 - darkness) * 0.15}%)`
    }
    if (darkness > 40) {
      return `hsl(${355 - tannin * 0.05}, ${65 + intensity * 0.2}%, ${35 + (100 - darkness) * 0.1}%)`
    }
    return `hsl(358, ${55 + intensity * 0.3}%, ${45 + (100 - darkness) * 0.15}%)`
  }

  const fillLevel = 35 + (intensity * 0.3)
  
  // More bubbles with better animation for sparkling
  const bubbles = useMemo(() => {
    const count = Math.floor(fizziness / 8)
    return Array.from({ length: count }, (_, i) => ({
      id: i,
      left: 35 + Math.random() * 30,
      delay: Math.random() * 3,
      size: 1.5 + Math.random() * 2.5,
      duration: 1.5 + Math.random() * 2,
    }))
  }, [fizziness])

  // Foam bubbles at the surface
  const foamBubbles = useMemo(() => {
    if (fizziness < 30) return []
    const count = Math.floor(fizziness / 15)
    return Array.from({ length: count }, (_, i) => ({
      id: i,
      left: 40 + (i * 5) + Math.random() * 3,
      size: 2 + Math.random() * 2,
      delay: Math.random() * 0.5,
    }))
  }, [fizziness])

  const wineColor = getWineColor()
  const rimOpacity = 0.3 + (acidity / 100) * 0.4
  
  // Calculate wine surface position
  const wineSurfaceY = 25 + (glassPath.fillTop - 25) * (1 - fillLevel / 100)

  return (
    <div className={cn("relative flex flex-col items-center", className)}>
      {/* Fixed height container for the SVG so tags don't affect it */}
      <div className="relative w-full flex-1 min-h-0 flex items-center justify-center">
        <svg
          viewBox="0 0 120 170"
          className="w-full h-full max-w-[180px]"
          style={{ filter: "drop-shadow(0 4px 12px rgba(0,0,0,0.1))" }}
        >
        <defs>
          <linearGradient id="wineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={wineColor} stopOpacity={rimOpacity} />
            <stop offset="30%" stopColor={wineColor} stopOpacity={0.9} />
            <stop offset="100%" stopColor={wineColor} stopOpacity={1} />
          </linearGradient>
          
          <linearGradient id="glassReflection" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="white" stopOpacity="0" />
            <stop offset="40%" stopColor="white" stopOpacity="0.12" />
            <stop offset="60%" stopColor="white" stopOpacity="0.12" />
            <stop offset="100%" stopColor="white" stopOpacity="0" />
          </linearGradient>

          <clipPath id="glassClip">
            <path d={glassPath.clipPath} />
          </clipPath>
        </defs>

        {/* Glass outline - uses transition for smooth morphing */}
        <path
          d={glassPath.path}
          fill="none"
          stroke="#E2D9CD"
          strokeWidth="1.5"
          style={{ transition: "d 0.4s ease-out" }}
        />

        {/* Wine fill area */}
        <g clipPath="url(#glassClip)">
          <rect
            x="10"
            y={wineSurfaceY}
            width="100"
            height={glassPath.baseY - wineSurfaceY + 10}
            fill="url(#wineGradient)"
            style={{ transition: "y 0.5s ease-out, height 0.5s ease-out" }}
          />
          
          {/* Wine surface curve */}
          <ellipse
            cx="60"
            cy={wineSurfaceY}
            rx={glassPath.rimWidth * 0.8}
            ry="3"
            fill={wineColor}
            opacity={0.5}
            style={{ transition: "cy 0.5s ease-out, rx 0.4s ease-out" }}
          />

          {/* Rising bubbles for sparkling */}
          {bubbles.map((bubble) => {
            // Start bubbles at bottom of wine, animate upward
            const startY = glassPath.fillTop - 10
            const endY = wineSurfaceY + 5
            const travelDistance = startY - endY
            
            return (
              <circle
                key={bubble.id}
                cx={bubble.left + 10}
                r={bubble.size}
                fill="white"
                style={{
                  cy: startY,
                  opacity: 0,
                  animation: `bubbleRise ${bubble.duration}s ease-out infinite`,
                  animationDelay: `${bubble.delay}s`,
                  // Use custom property for dynamic travel distance
                  ['--travel' as string]: `-${travelDistance}px`,
                }}
              />
            )
          })}

          {/* Foam bubbles at surface */}
          {foamBubbles.map((bubble) => (
            <circle
              key={`foam-${bubble.id}`}
              cx={bubble.left}
              cy={wineSurfaceY + 2}
              r={bubble.size}
              fill="white"
              opacity="0.5"
              style={{
                animation: `fizz-wobble 1s ease-in-out infinite`,
                animationDelay: `${bubble.delay}s`,
                transition: "cy 0.5s ease-out",
              }}
            />
          ))}
        </g>

        {/* Glass reflection overlay */}
        <path
          d={glassPath.path}
          fill="url(#glassReflection)"
          opacity="0.5"
          className="pointer-events-none"
          style={{ transition: "d 0.4s ease-out" }}
        />

        {/* Rim highlight */}
        <ellipse
          cx="60"
          cy="20"
          rx={glassPath.rimWidth}
          ry="3"
          fill="none"
          stroke="#E2D9CD"
          strokeWidth="1"
          opacity="0.5"
          style={{ transition: "rx 0.4s ease-out" }}
        />
        </svg>
      </div>

      {/* Structure indicators - fixed at bottom, don't affect glass size */}
      <div className="h-8 flex flex-wrap justify-center items-center gap-1.5 text-xs shrink-0">
        {intensity > 60 && (
          <span className="px-2 py-0.5 rounded-full bg-primary/10 text-primary transition-all">Bold</span>
        )}
        {tannin > 60 && (
          <span className="px-2 py-0.5 rounded-full bg-primary/10 text-primary transition-all">Tannic</span>
        )}
        {acidity > 70 && (
          <span className="px-2 py-0.5 rounded-full bg-accent/10 text-accent transition-all">Crisp</span>
        )}
        {sweetness > 30 && (
          <span className="px-2 py-0.5 rounded-full bg-secondary/30 text-secondary-foreground transition-all">Sweet</span>
        )}
        {fizziness > 50 && (
          <span className="px-2 py-0.5 rounded-full bg-secondary/30 text-secondary-foreground transition-all">Sparkling</span>
        )}
      </div>
    </div>
  )
}
