# MetAI Kubernetes Deployment

This directory contains Kubernetes manifests for deploying MetAI on a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured to access your cluster
- Ingress controller installed (e.g., nginx-ingress)
- Sufficient cluster resources (at least 8GB RAM, 4 CPUs)
- Docker images built and pushed to a registry

## Quick Start

### 1. Update Configuration

**Edit `secrets.yaml`:**
```bash
# Update with your actual API keys
vi k8s/secrets.yaml
```

Required secrets:
- `OPENAI_API_KEY`
- `PAGEINDEX_API_KEY`
- `OPIK_API_KEY`
- `OPIK_WORKSPACE`
- `JWT_SECRET_KEY`

**Edit `ingress.yaml`:**
```bash
# Replace domain names
vi k8s/ingress.yaml
# Change: metai.yourdomain.com to your actual domain
```

### 2. Build and Push Docker Images

```bash
# Build API image
cd app
docker build -t your-registry/metai-api:latest .
docker push your-registry/metai-api:latest

# Build Web image
cd ../web_app
docker build -t your-registry/metai-web:latest .
docker push your-registry/metai-web:latest
```

### 3. Update Image References

Edit deployment files to use your registry:
```bash
# In api-deployment.yaml
image: your-registry/metai-api:latest

# In web-deployment.yaml
image: your-registry/metai-web:latest
```

### 4. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create ConfigMap and Secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Create PersistentVolumeClaims
kubectl apply -f k8s/pvc.yaml

# Deploy MongoDB and Redis
kubectl apply -f k8s/mongodb-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=mongodb -n metai --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n metai --timeout=300s

# Deploy API and Web
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/web-deployment.yaml

# Create Ingress
kubectl apply -f k8s/ingress.yaml

# (Optional) Apply autoscaling
kubectl apply -f k8s/hpa.yaml
```

### 5. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n metai

# Check services
kubectl get svc -n metai

# Check ingress
kubectl get ingress -n metai

# View logs
kubectl logs -f deployment/api -n metai
kubectl logs -f deployment/web -n metai
```

## Architecture

```
                     Internet
                        |
                   [Ingress]
                    /      \
            [Web Service]  [API Service]
                 |              |
            [Web Pods]     [API Pods]
            (2+ replicas)  (2+ replicas)
                              /    \
                    [MongoDB]      [Redis]
                    (StatefulSet)  (1 replica)
```

## Resource Requirements

**Minimum per pod:**
- API: 2GB RAM, 1 CPU
- Web: 512MB RAM, 250m CPU
- MongoDB: 512MB RAM, 250m CPU
- Redis: 256MB RAM, 100m CPU

**Recommended for production:**
- API: 4GB RAM, 2 CPU (2-10 replicas with HPA)
- Web: 1GB RAM, 1 CPU (2-10 replicas with HPA)
- MongoDB: 2GB RAM, 1 CPU
- Redis: 1GB RAM, 500m CPU

## Storage

- **MongoDB**: 10Gi persistent volume
- **Redis**: 5Gi persistent volume
- **API Logs**: 5Gi persistent volume
- **Temp Uploads**: EmptyDir (ephemeral)

## Scaling

### Manual Scaling
```bash
# Scale API
kubectl scale deployment api --replicas=5 -n metai

# Scale Web
kubectl scale deployment web --replicas=5 -n metai
```

### Auto Scaling (HPA)
The `hpa.yaml` configures automatic scaling based on CPU/memory:
- Min replicas: 2
- Max replicas: 10
- Target CPU: 70%
- Target Memory: 80%

## Monitoring

### Check Pod Status
```bash
kubectl get pods -n metai -w
```

### View Logs
```bash
# API logs
kubectl logs -f deployment/api -n metai

# Web logs
kubectl logs -f deployment/web -n metai

# MongoDB logs
kubectl logs -f deployment/mongodb -n metai

# Redis logs
kubectl logs -f deployment/redis -n metai
```

### Exec into Pod
```bash
# API pod
kubectl exec -it deployment/api -n metai -- /bin/bash

# Web pod
kubectl exec -it deployment/web -n metai -- /bin/sh
```

## Troubleshooting

### Pods not starting
```bash
# Check pod events
kubectl describe pod <pod-name> -n metai

# Check if secrets are created
kubectl get secrets -n metai

# Check if configmap is created
kubectl get configmap -n metai
```

### Database connection issues
```bash
# Test MongoDB connectivity
kubectl run -it --rm debug --image=mongo:7 --restart=Never -n metai -- mongosh mongodb://mongodb-service:27017

# Test Redis connectivity
kubectl run -it --rm debug --image=redis:7-alpine --restart=Never -n metai -- redis-cli -h redis-service
```

### Ingress not working
```bash
# Check ingress controller is installed
kubectl get pods -n ingress-nginx

# Check ingress configuration
kubectl describe ingress metai-ingress -n metai

# Check service endpoints
kubectl get endpoints -n metai
```

### Image pull errors
```bash
# Check if images are accessible
docker pull your-registry/metai-api:latest

# If using private registry, create imagePullSecret
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n metai

# Add imagePullSecrets to deployment
# Add this under spec.template.spec:
#   imagePullSecrets:
#   - name: regcred
```

## Updating Deployment

### Update API
```bash
# Build new image
cd app
docker build -t your-registry/metai-api:v1.1.0 .
docker push your-registry/metai-api:v1.1.0

# Update deployment
kubectl set image deployment/api api=your-registry/metai-api:v1.1.0 -n metai

# Or apply updated manifest
kubectl apply -f k8s/api-deployment.yaml
```

### Update Web
```bash
# Build new image
cd web_app
docker build -t your-registry/metai-web:v1.1.0 .
docker push your-registry/metai-web:v1.1.0

# Update deployment
kubectl set image deployment/web web=your-registry/metai-web:v1.1.0 -n metai
```

### Rollback
```bash
# View rollout history
kubectl rollout history deployment/api -n metai

# Rollback to previous version
kubectl rollout undo deployment/api -n metai

# Rollback to specific revision
kubectl rollout undo deployment/api --to-revision=2 -n metai
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/

# Or delete namespace (deletes everything)
kubectl delete namespace metai
```

## Security Best Practices

1. **Use secrets for sensitive data** - Never commit actual secrets to git
2. **Enable RBAC** - Create service accounts with minimal permissions
3. **Network policies** - Restrict pod-to-pod communication
4. **TLS/HTTPS** - Use cert-manager for automatic SSL certificates
5. **Image scanning** - Scan Docker images for vulnerabilities
6. **Resource limits** - Always set resource requests and limits
7. **Non-root containers** - Run containers as non-root user (web already does)
8. **Pod security policies** - Enforce security standards

## Production Checklist

- [ ] Update all secrets with production values
- [ ] Configure TLS/HTTPS in ingress
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure log aggregation (ELK/Loki)
- [ ] Set up backup for MongoDB data
- [ ] Configure network policies
- [ ] Enable pod disruption budgets
- [ ] Set up alerts for pod/service failures
- [ ] Configure resource quotas per namespace
- [ ] Test disaster recovery procedures

## Support

For issues or questions:
- GitHub: https://github.com/[your-username]/MetAI
- Documentation: See main README.md
