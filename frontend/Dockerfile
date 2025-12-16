# Use Node.js base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json yarn.lock* ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy the rest of the application
COPY . .

# Build the application
RUN yarn build

# Expose port
EXPOSE 3000

# Run the application
CMD ["yarn", "serve", "--host", "0.0.0.0", "--port", "3000"]