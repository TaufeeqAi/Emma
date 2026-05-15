/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        nhs: {
          blue:       "#003087",
          "blue-mid": "#0072CE",
          "blue-lt":  "#41B6E6",
          green:      "#007F3B",
          red:        "#DA291C",
          yellow:     "#FFB81C",
          white:      "#FFFFFF",
          grey:       "#425563",
          "grey-lt":  "#E8EDEE",
        },
      },
      fontFamily: {
        sans: ["Frutiger", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};