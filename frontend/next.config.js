/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Proxy /api/* → FastAPI backend to avoid CORS in dev
  async rewrites() {
    return [
      {
        source:      "/api/:path*",
        destination: `${process.env.BACKEND_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;