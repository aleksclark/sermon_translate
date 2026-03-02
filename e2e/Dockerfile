FROM mcr.microsoft.com/playwright:v1.52.0-noble
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && corepack prepare pnpm@latest --activate && pnpm install --frozen-lockfile
COPY . .
CMD ["pnpm", "test"]
