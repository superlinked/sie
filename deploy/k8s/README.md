# Kubernetes manifests

The SIE Helm chart is the sole production source for Kubernetes resources,
including the per-physical-lane KEDA `ScaledObject` contract. Use
[`deploy/helm/sie-cluster`](../helm/sie-cluster/) rather than maintaining a
second raw autoscaling manifest here.

This directory remains checked in because the public-repository sync contract
copies `deploy/k8s` as a published path.
