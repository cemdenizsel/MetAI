/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required for Docker standalone build
  output: 'standalone',

  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
      },
    ],
  },
  env: {
    API_URL: process.env.NODE_ENV === 'development'
      ? (process.env.API_URL || 'http://localhost:8084')
      : (process.env.API_URL || 'https://be.meetingai.info'),
  },
};

module.exports = nextConfig;
