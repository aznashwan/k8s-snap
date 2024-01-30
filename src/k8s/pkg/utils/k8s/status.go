package k8s

import (
	"context"
	"fmt"

	"github.com/canonical/k8s/pkg/utils"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// WaitApiServerReady waits until the kube-apiserver becomes available.
func WaitApiServerReady(ctx context.Context, client *k8sClient) error {
	return utils.WaitUntilReady(ctx, func() (bool, error) {
		_, err := client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		// We want to retry if an error occurs (=API server not ready)
		// returning the error would abort, thus checking for nil
		return err == nil, nil
	})
}

// ClusterReady checks the status of all nodes in the Kubernetes cluster.
// If at least one node is in READY state it will return true.
func ClusterReady(ctx context.Context, client *k8sClient) (bool, error) {
	nodes, err := client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("failed to get cluster nodes: %v", err)
	}

	for _, node := range nodes.Items {
		for _, condition := range node.Status.Conditions {
			if condition.Type == "Ready" {
				if condition.Status == v1.ConditionTrue {
					// At least one node is ready.
					// That's enough for the cluster to operate.
					return true, nil
				}
			}
		}
	}

	return false, nil
}