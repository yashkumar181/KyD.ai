/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          primary: '#0a0a0a',
          surface: '#111111',
          elevated: '#1a1a1a',
        },
        border: {
          subtle: '#2a2a2a',
          active: '#3a3a3a',
        },
        accent: {
          primary: '#6366f1',
          success: '#22c55e',
          warning: '#f59e0b',
          danger: '#ef4444',
        },
        text: {
          primary: '#f5f5f5',
          secondary: '#a1a1aa',
          muted: '#52525b',
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}