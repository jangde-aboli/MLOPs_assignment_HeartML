#!/usr/bin/env bash
# Build the image and deploy API + monitoring to the current kubectl context
# (Docker Desktop Kubernetes or Minikube).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=heart-api:1.0.0

if [[ ! -f models/heart_model.joblib ]]; then
  echo ">> No model found, training one"
  .venv/Scripts/python -m heartml.train 2>/dev/null || python -m heartml.train
fi

if kubectl config current-context | grep -q minikube; then
  echo ">> Building inside Minikube's Docker daemon"
  eval "$(minikube docker-env)"
fi
docker build -t "$IMAGE" .

# Docker Desktop's kind-based cluster has its own containerd, so a locally
# built image is invisible to it. Import it through a node debug pod.
if kubectl get nodes -o name | grep -q "desktop-control-plane"; then
  echo ">> Loading $IMAGE into the kind node"
  tar="$(mktemp)"
  docker save "$IMAGE" -o "$tar"
  MSYS_NO_PATHCONV=1 kubectl debug node/desktop-control-plane -i --quiet \
    --image=busybox:1.36 --profile=sysadmin -- \
    chroot /host ctr -n k8s.io images import - < "$tar"
  rm -f "$tar"
  kubectl get pods -o name | grep node-debugger | xargs -r kubectl delete --wait=false
fi

kubectl apply -f k8s/api.yaml
kubectl -n heart-mlops create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring.yaml

kubectl -n heart-mlops rollout status deployment/heart-api --timeout=120s
kubectl -n heart-mlops get pods,svc,ingress

cat <<EOF

API service is type LoadBalancer:
  Docker Desktop : http://localhost/health
  Minikube       : run 'minikube tunnel' in another terminal, then use the EXTERNAL-IP above
Prometheus : http://localhost:9090  (Minikube: minikube service prometheus -n heart-mlops)
Grafana    : http://localhost:3000  (Minikube: minikube service grafana -n heart-mlops)
EOF
