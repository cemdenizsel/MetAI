/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
      },
    ],
  },
  env: {
    API_URL: process.env.API_URL || 'http://localhost:8084',
  },
};

module.exports = nextConfig;
