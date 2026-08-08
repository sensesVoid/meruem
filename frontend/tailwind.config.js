/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        meruem: {
          bg: "#0b0f14",
          panel: "#11161d",
          border: "#1f2732",
          accent: "#38bdf8",
          gold: "#eab308",
          yellow: "#facc15",
          green: "#22c55e",
          red: "#ef4444",
          text: "#e2e8f0",
          muted: "#7c8aa0",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};