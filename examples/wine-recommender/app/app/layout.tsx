import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Sommelier — Wine Recommender',
  description: 'Discover wines tailored to your palate',
  applicationName: 'Sommelier',
  icons: {
    icon: '/icon.svg',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  )
}
