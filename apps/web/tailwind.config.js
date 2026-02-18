module.exports = {
  darkMode: ["class"],
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bg: "rgb(var(--bg))",
        panel: "rgb(var(--panel))",
        border: "rgb(var(--border))",
        text: "rgb(var(--text))",
        muted: "rgb(var(--muted))",
        brand: "rgb(var(--brand))",
      },
      boxShadow: {
        soft: "0 10px 30px rgba(0,0,0,.08)",
      },
    },
  },
  plugins: [],
};
