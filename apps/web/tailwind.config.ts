import type { Config } from "tailwindcss"

const config: Config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', '"Avenir Next"', "sans-serif"],
        serif: ['"Iowan Old Style"', '"Source Serif Pro"', "serif"],
        mono: ['"IBM Plex Mono"', "monospace"],
      },
      colors: {
        background: "hsl(var(--background) / <alpha-value>)",
        foreground: "hsl(var(--foreground) / <alpha-value>)",
        border: "hsl(var(--border) / <alpha-value>)",
        input: "hsl(var(--input) / <alpha-value>)",
        ring: "hsl(var(--ring) / <alpha-value>)",
        card: "hsl(var(--card) / <alpha-value>)",
        "card-foreground": "hsl(var(--card-foreground) / <alpha-value>)",
        popover: "hsl(var(--popover) / <alpha-value>)",
        "popover-foreground": "hsl(var(--popover-foreground) / <alpha-value>)",
        primary: {
          DEFAULT: "hsl(var(--primary) / <alpha-value>)",
          foreground: "hsl(var(--primary-foreground) / <alpha-value>)",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary) / <alpha-value>)",
          foreground: "hsl(var(--secondary-foreground) / <alpha-value>)",
        },
        muted: {
          DEFAULT: "hsl(var(--muted) / <alpha-value>)",
          foreground: "hsl(var(--muted-foreground) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "hsl(var(--accent) / <alpha-value>)",
          foreground: "hsl(var(--accent-foreground) / <alpha-value>)",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive) / <alpha-value>)",
          foreground: "hsl(var(--destructive-foreground) / <alpha-value>)",
        },
        verdict: {
          supported: "hsl(var(--verdict-supported) / <alpha-value>)",
          partial: "hsl(var(--verdict-partial) / <alpha-value>)",
          overstated: "hsl(var(--verdict-overstated) / <alpha-value>)",
          ambiguous: "hsl(var(--verdict-ambiguous) / <alpha-value>)",
          unsupported: "hsl(var(--verdict-unsupported) / <alpha-value>)",
          unverified: "hsl(var(--verdict-unverified) / <alpha-value>)",
        },
        surface: {
          "0": "hsl(var(--surface-0) / <alpha-value>)",
          "1": "hsl(var(--surface-1) / <alpha-value>)",
          "2": "hsl(var(--surface-2) / <alpha-value>)",
          "3": "hsl(var(--surface-3) / <alpha-value>)",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      boxShadow: {
        panel: "0 0 0 1px hsl(var(--border) / 0.5), 0 12px 24px hsl(0 0% 0% / 0.35)",
      },
    },
  },
  plugins: [],
}

export default config
