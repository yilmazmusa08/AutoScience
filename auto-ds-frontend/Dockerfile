FROM node:18-alpine as build

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install project dependencies (use yarn if you have a yarn.lock file)
RUN npm install

# Copy the entire project directory into the container
COPY . .

RUN npm run build


FROM nginx:alpine

# Remove the default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Create a directory to hold the frontend build
RUN mkdir -p /usr/share/nginx/html

# Copy the built frontend files into the container
COPY --from=build /app/build /usr/share/nginx/html

# Copy your custom Nginx configuration to the container
COPY ./nginx/nginx.conf /etc/nginx/conf.d/

# Expose port 80 for Nginx
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]