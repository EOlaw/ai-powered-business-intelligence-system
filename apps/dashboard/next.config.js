/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  env: {
    NEXT_PUBLIC_GATEWAY_URL: process.env.NEXT_PUBLIC_GATEWAY_URL ?? 'http://localhost:3000',
  },
  webpack(config) {
    // Allow .js imports to resolve .tsx/.ts files (TypeScript ESM convention)
    config.resolve.extensionAlias = {
      '.js': ['.tsx', '.ts', '.js'],
    };
    return config;
  },
};

module.exports = nextConfig;
