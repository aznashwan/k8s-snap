package metrics_server_test

import (
	"context"
	"testing"

	"github.com/canonical/k8s/pkg/client/helm"
	helmmock "github.com/canonical/k8s/pkg/client/helm/mock"
	metrics_server "github.com/canonical/k8s/pkg/k8sd/features/metrics-server"
	"github.com/canonical/k8s/pkg/k8sd/types"
	snapmock "github.com/canonical/k8s/pkg/snap/mock"
	"github.com/canonical/k8s/pkg/utils"
	. "github.com/onsi/gomega"
)

func TestApplyMetricsServer(t *testing.T) {
	for _, tc := range []struct {
		name        string
		config      types.MetricsServer
		expectState helm.State
	}{
		{
			name: "Enable",
			config: types.MetricsServer{
				Enabled: utils.Pointer(true),
			},
			expectState: helm.StatePresent,
		},
		{
			name: "Disable",
			config: types.MetricsServer{
				Enabled: utils.Pointer(false),
			},
			expectState: helm.StateDeleted,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := NewWithT(t)
			h := &helmmock.Mock{}
			s := &snapmock.Snap{
				Mock: snapmock.Mock{
					HelmClient: h,
				},
			}

			err := metrics_server.ApplyMetricsServer(context.Background(), s, tc.config, nil)
			g.Expect(err).ToNot(HaveOccurred())

			g.Expect(h.ApplyCalledWith).To(ConsistOf(SatisfyAll(
				HaveField("Chart.Name", Equal("metrics-server")),
				HaveField("Chart.Namespace", Equal("kube-system")),
				HaveField("State", Equal(tc.expectState)),
			)))
		})
	}

	t.Run("Annotations", func(t *testing.T) {
		g := NewWithT(t)
		h := &helmmock.Mock{}
		s := &snapmock.Snap{
			Mock: snapmock.Mock{
				HelmClient: h,
			},
		}

		cfg := types.MetricsServer{
			Enabled: utils.Pointer(true),
		}
		annotations := types.Annotations{
			"k8sd/v1alpha1/metrics-server/image-repo": "custom-image",
			"k8sd/v1alpha1/metrics-server/image-tag":  "custom-tag",
		}

		err := metrics_server.ApplyMetricsServer(context.Background(), s, cfg, annotations)
		g.Expect(err).To(BeNil())
		g.Expect(h.ApplyCalledWith).To(ConsistOf(HaveField("Values", HaveKeyWithValue("image", SatisfyAll(
			HaveKeyWithValue("repository", "custom-image"),
			HaveKeyWithValue("tag", "custom-tag"),
		)))))
	})
}
