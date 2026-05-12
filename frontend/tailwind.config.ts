import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          50:  "#e0f7ff",
          100: "#b3ecff",
          200: "#80deff",
          300: "#4dd0ff",
          400: "#26c5ff",
          500: "#00baff",
          600: "#00a8e8",
          700: "#0090cc",
          800: "#0078b0",
          900: "#005580",
        },
        surface: {
          DEFAULT: "#0D1117",
          1: "#161B22",
          2: "#1C2333",
          3: "#252B3B",
          4: "#2D3548",
        },
        danger:  { DEFAULT: "#EF4444", light: "#FCA5A5" },
        warning: { DEFAULT: "#F59E0B", light: "#FCD34D" },
        success: { DEFAULT: "#10B981", light: "#6EE7B7" },
        muted:   "#8892A4",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      backgroundImage: {
        "hero-gradient": "linear-gradient(135deg, #0D1117 0%, #1a1f2e 50%, #0D2137 100%)",
        "brand-gradient": "linear-gradient(90deg, #00D4FF, #0072FF, #a855f7)",
        "card-gradient": "linear-gradient(135deg, #1a1f2e, #252B3B)",
      },
      borderRadius: { xl: "1rem", "2xl": "1.25rem", "3xl": "1.5rem" },
      animation: {
        "fade-in":    "fadeIn 0.3s ease-out",
        "slide-up":   "slideUp 0.4s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeIn:  { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp: { from: { opacity: "0", transform: "translateY(12px)" }, to: { opacity: "1", transform: "translateY(0)" } },
      },
    },
  },
  plugins: [],
} satisfies Config;
