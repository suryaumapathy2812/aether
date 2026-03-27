import { Outlet, createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/integrations/$name")({
  component: IntegrationDetailLayout,
});

function IntegrationDetailLayout() {
  return <Outlet />;
}
