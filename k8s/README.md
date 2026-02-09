# MetAI Kubernetes Deployment

This directory contains all Kubernetes manifests for deploying MetAI to a Kubernetes cluster.

## Directory Structure

- **namespace.yaml** - MetAI namespace
- **configmap.yaml** - Configuration variables
- **secrets.yaml** - Sensitive data (update with real values)
- **pvc.yaml** - Persistent volumes for MongoDB and Redis
- **mongodb-deployment.yaml** - MongoDB database deployment
- **redis-deployment.yaml** - Redis cache deployment
- **api-deployment.yaml** - FastAPI backend deployment
- **web-deployment.yaml** - Next.js frontend deployment
- **ingress.yaml** - Ingress configuration for routing
- **hpa.yaml** - Horizontal Pod Autoscaler for auto-scaling

## Prerequisites

1. Kubernetes cluster (1.20+)
2. kubectl configured to access your cluster
3. NGINX Ingress Controller installed
4. cert-manager for SSL/TLS certificates (optional but recommended)
5. Docker images pushed to a container registry

## Setup Instructions

### 1. Build and Push Docker Images

```bash
# Build backend image
docker build -t your-registry/metai-api:latest ./app
docker push your-registry/metai-api:latest

# Build frontend image
docker build -t your-registry/metai-web:latest ./web_app
docker push your-registry/metai-web:latest
```

### 2. Update Image Registries

Edit the following files and replace `your-registry` with your actual registry:
- `api-deployment.yaml` - Update image: `your-registry/metai-api:latest`
- `web-deployment.yaml` - Update image: `your-registry/metai-web:latest`

### 3. Create Kubernetes Secrets from .env File

Instead of manually editing `secrets.yaml`, use the automated script to load secrets from your `.env` file:

```bash
cd k8s

# Simple method - loads all variables from .env
./create-secrets-simple.sh metai ../.env

# Or advanced method - loads only sensitive variables
./create-secrets.sh metai
```

For detailed instructions, see [SECRETS_MANAGEMENT.md](./SECRETS_MANAGEMENT.md)

### 4. Install NGINX Ingress Controller (if not already installed)

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install nginx-ingress ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace
```

### 5. Install cert-manager for SSL/TLS (recommended)

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 6. Deploy to Kubernetes

```bash
# Deploy all manifests in order
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Create secrets from .env file (REQUIRED!)
cd k8s && ./create-secrets-simple.sh metai ../.env && cd ..

# Deploy remaining resources
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/mongodb-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/web-deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```

Or deploy everything at once (after creating secrets):
```bash
cd k8s && ./create-secrets-simple.sh metai ../.env && kubectl apply -f . && cd ..
```

### 7. Configure DNS

Point your domain DNS records to your Kubernetes cluster's load balancer:
- `meetingai.info` → Load Balancer IP
- `be.meetingai.info` → Load Balancer IP

Get the Load Balancer IP:
```bash
kubectl get svc -n ingress-nginx
```

## Verification

### Check Namespace and Resources

```bash
# Check namespace
kubectl get ns | grep metai

# Check all resources in metai namespace
kubectl get all -n metai

# Check ingress
kubectl get ingress -n metai
kubectl describe ingress metai-ingress -n metai

# Check certificates
kubectl get certificate -n metai
```

### Check Pod Status

```bash
# Watch pods starting
kubectl get pods -n metai -w

# View logs
kubectl logs -n metai -l app=api
kubectl logs -n metai -l app=web
kubectl logs -n metai -l app=mongodb
kubectl logs -n metai -l app=redis
```

### Test API Connectivity

```bash
# Test backend API
curl https://be.meetingai.info/health

# Test frontend
curl https://meetingai.info/
```

## Environment Variables

### Configurable via ConfigMap

- `API_HOST` - API server host (default: 0.0.0.0)
- `API_PORT` - API server port (default: 8084)
- `MONGODB_HOST` - MongoDB host
- `MONGODB_PORT` - MongoDB port
- `MONGODB_DATABASE` - Database name
- `REDIS_HOST` - Redis host
- `REDIS_PORT` - Redis port
- `NEXT_PUBLIC_API_URL` - Frontend API URL (set to https://be.meetingai.info)
- `NODE_ENV` - Node environment (production)

### Secrets

- `MONGODB_CONNECTION_STRING` - MongoDB connection string
- Add other sensitive data as needed

## Scaling

### Manual Scaling

```bash
# Scale API deployment
kubectl scale deployment api -n metai --replicas=3

# Scale web deployment
kubectl scale deployment web -n metai --replicas=3
```

### Auto-scaling (HPA)

The `hpa.yaml` automatically scales pods based on:
- CPU utilization (70% threshold)
- Memory utilization (80% threshold)
- Min replicas: 2
- Max replicas: 5

Monitor autoscaling:
```bash
kubectl get hpa -n metai
kubectl describe hpa api-hpa -n metai
```

## Monitoring and Troubleshooting

### View Pod Events

```bash
kubectl describe pod <pod-name> -n metai
```

### Check Resource Usage

```bash
kubectl top pods -n metai
kubectl top nodes
```

### View Recent Logs

```bash
kubectl logs -n metai -l app=api --tail=100 -f
kubectl logs -n metai -l app=web --tail=100 -f
```

## Cleanup

To remove all MetAI resources:

```bash
kubectl delete namespace metai
```

## Storage Considerations

Current setup uses:
- MongoDB: 10Gi PVC
- Redis: 5Gi PVC

To change storage sizes, edit `pvc.yaml` and update the `storage` field, then reapply.

## Production Considerations

1. **Update Container Registry**: Replace `your-registry` with your actual Docker registry
2. **Resource Limits**: Adjust CPU/memory requests and limits based on your cluster
3. **Replicas**: Adjust `replicas` in deployments based on expected load
4. **TLS**: Update cert-manager email in ClusterIssuer
5. **DNS**: Configure proper DNS records for your domains
6. **Persistent Data**: For production, consider managed database services (AWS RDS, Google Cloud SQL, etc.)
7. **Backup**: Set up regular backups for MongoDB data
8. **Monitoring**: Add Prometheus and Grafana for monitoring
9. **Logging**: Consider ELK stack or similar for centralized logging

## Useful Commands

```bash
# Get all resources in namespace
kubectl get all -n metai

# Watch deployment progress
kubectl rollout status deployment/api -n metai

# View deployment YAML
kubectl get deployment api -n metai -o yaml

# Port forward for local testing
kubectl port-forward -n metai svc/api 8084:8084
kubectl port-forward -n metai svc/web 3000:3000

# Delete and redeploy
kubectl rollout restart deployment/api -n metai
```
