# Claude Multi-Agent Deployment Guide

This directory contains all deployment configurations for the Claude Multi-Agent Coordination System, including Docker containers, Kubernetes manifests, Helm charts, and multi-region deployment support.

## Directory Structure

```
deployment/
├── docker/                     # Docker configurations
│   ├── Dockerfile.option-a     # File-based coordination
│   ├── Dockerfile.option-b     # MCP Server
│   ├── Dockerfile.option-c     # External Orchestrator
│   ├── docker-compose.yml      # Local development stack
│   └── prometheus.yml          # Prometheus configuration
├── kubernetes/                 # Kubernetes manifests
│   ├── namespace.yaml          # Namespace definition
│   ├── configmap.yaml          # Configuration maps
│   ├── secrets.yaml            # Secrets (placeholders)
│   ├── storage.yaml            # Storage classes and PVCs
│   ├── rbac.yaml               # RBAC configuration
│   ├── deployment-option-*.yaml # Deployments for each option
│   ├── hpa.yaml                # Horizontal Pod Autoscaler
│   ├── ingress.yaml            # Ingress and Load Balancer
│   ├── istio.yaml              # Istio service mesh config
│   ├── redis.yaml              # Redis for distributed state
│   ├── multi-region.yaml       # Multi-region deployment
│   ├── resource-quotas.yaml    # Resource quotas and limits
│   └── kustomization.yaml      # Kustomize configuration
├── helm/                       # Helm chart
│   ├── Chart.yaml              # Chart definition
│   ├── values.yaml             # Default values
│   └── templates/              # Helm templates
└── terraform/                  # Infrastructure as Code (future)
```

## Quick Start

### Local Development with Docker Compose

```bash
# Build and start all services
cd deployment/docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Kubernetes Deployment

```bash
# Apply all manifests
cd deployment/kubernetes
kubectl apply -k .

# Or apply individually
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f storage.yaml
kubectl apply -f rbac.yaml
kubectl apply -f redis.yaml
kubectl apply -f deployment-option-c.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
```

### Helm Deployment

```bash
# Add dependencies
cd deployment/helm
helm dependency update

# Install the chart
helm install claude-multiagent . \
  --namespace claude-multiagent \
  --create-namespace \
  --set secrets.anthropicApiKey=your-api-key

# Upgrade
helm upgrade claude-multiagent . --namespace claude-multiagent

# Uninstall
helm uninstall claude-multiagent --namespace claude-multiagent
```

## Scalability Features

### 1. Multi-Node Orchestrator Support (adv-scale-001)

The orchestrator deployment supports multiple replicas with leader election:

```yaml
# deployment-option-c.yaml
spec:
  replicas: 3
```

Redis is used for distributed state coordination between nodes.

### 2. Kubernetes Deployment Manifests (adv-scale-002)

Complete K8s manifests in `kubernetes/` directory including:
- Deployments with anti-affinity rules
- Services (ClusterIP and LoadBalancer)
- ConfigMaps and Secrets
- PersistentVolumeClaims

### 3. Docker Containerization (adv-scale-003)

Multi-stage Dockerfiles for each option:
- `Dockerfile.option-a` - Python-based file coordination
- `Dockerfile.option-b` - Node.js MCP server
- `Dockerfile.option-c` - Python orchestrator

### 4. Horizontal Scaling Configuration (adv-scale-004)

HPA configuration in `hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          averageUtilization: 60
```

### 5. Load Balancer Integration (adv-scale-005)

Multiple ingress options in `ingress.yaml`:
- NGINX Ingress Controller
- AWS ALB Ingress
- GCP GKE Ingress
- Direct LoadBalancer Service

### 6. Service Mesh Compatibility (adv-scale-006)

Istio configuration in `istio.yaml`:
- VirtualService for traffic routing
- DestinationRules for load balancing
- PeerAuthentication for mTLS
- AuthorizationPolicy for access control

### 7. Cloud-Native Storage Backends (adv-scale-007)

Storage classes in `storage.yaml`:
- GCP Persistent Disk (SSD)
- AWS EBS (gp3)
- Azure Disk (Premium_LRS)
- ReadWriteMany support for shared storage

### 8. Auto-Scaling Policies (adv-scale-008)

Advanced HPA with custom metrics:

```yaml
metrics:
  - type: Pods
    pods:
      metric:
        name: pending_tasks
      target:
        averageValue: "20"
behavior:
  scaleUp:
    stabilizationWindowSeconds: 0
  scaleDown:
    stabilizationWindowSeconds: 600
```

### 9. Multi-Region Deployment Support (adv-scale-009)

Configuration in `multi-region.yaml`:
- ServiceExport for multi-cluster service discovery
- MultiClusterIngress for global load balancing
- Cross-region state synchronization CronJob
- Topology-aware routing

### 10. Resource Quota Management (adv-scale-010)

Quotas in `resource-quotas.yaml`:
- Namespace-level resource quotas
- LimitRanges for default constraints
- PriorityClasses for workload prioritization
- PodDisruptionBudgets for availability

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COORDINATION_DIR` | Path to coordination data | `/data/.coordination` |
| `ORCHESTRATOR_MAX_WORKERS` | Maximum worker agents | `3` |
| `ORCHESTRATOR_TASK_TIMEOUT` | Task timeout in seconds | `600` |
| `REDIS_HOST` | Redis hostname | `redis-master` |
| `REDIS_PORT` | Redis port | `6379` |

### Secrets

Store sensitive data in Kubernetes secrets:

```bash
kubectl create secret generic coordinator-secrets \
  --namespace claude-multiagent \
  --from-literal=ANTHROPIC_API_KEY=your-key \
  --from-literal=REDIS_PASSWORD=your-password
```

## Monitoring

### Prometheus Metrics

The orchestrator exposes metrics on port 9090:
- `tasks_total` - Total tasks created
- `tasks_completed` - Completed tasks
- `tasks_failed` - Failed tasks
- `workers_active` - Active workers
- `task_duration_seconds` - Task execution time

### Grafana Dashboard

Import the dashboard from `docker/grafana-dashboard.json` or use the Helm chart with Grafana enabled.

## Production Checklist

- [ ] Configure proper resource limits
- [ ] Enable TLS for ingress
- [ ] Set up proper secrets management (e.g., Vault)
- [ ] Configure backup for PersistentVolumes
- [ ] Set up monitoring alerts
- [ ] Configure network policies
- [ ] Enable Istio mTLS (if using service mesh)
- [ ] Configure multi-region replication (if needed)
- [ ] Review and adjust HPA settings
- [ ] Set appropriate resource quotas

## Troubleshooting

### Check pod status
```bash
kubectl get pods -n claude-multiagent
kubectl describe pod <pod-name> -n claude-multiagent
```

### View logs
```bash
kubectl logs -f deployment/orchestrator -n claude-multiagent
```

### Check HPA status
```bash
kubectl get hpa -n claude-multiagent
kubectl describe hpa orchestrator-hpa -n claude-multiagent
```

### Redis connectivity
```bash
kubectl exec -it redis-master-0 -n claude-multiagent -- redis-cli ping
```
